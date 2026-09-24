"""Stage `normalize`: names and addresses -> several comparable representations.

Nothing is thrown away: each record gets a light form (`*_norm`), stripped forms
(`name_core`, `addr_core`) and extracted pieces (legal forms, domain, initials, state,
house numbers, unit, landmark). The model decides how much to trust each.

Name fields
  name_norm      ASCII, lowercase, junk removed, legal words canonical
  name_core      name_norm minus legal words / stopwords, tokens sorted (word-order proof)
  name_concat    core tokens in original order, no spaces (compare with domains)
  name_initials  first letter of each core token
  name_skel      consonant skeleton of each core token, sorted (typos, transliteration)
  name_legal     canonical legal forms, sorted, space-joined
  name_domain    label of a website-style name ("vivekcharitabletrust.com" -> "vivekcharitabletrust")
  name_translit  name was written in an Indic script

Address fields
  addr_norm      cleaned components in original order (care-of and landmarks removed)
  addr_core      unique tokens sorted, state components and markers removed
  addr_state     explicit state/region, else one learned from training addresses
  addr_house     numbers introduced by a marker (plot/no/#/door/...) or leading street number
  addr_numbers   every other number (units excluded), leading zeros stripped
  addr_unit      unit / floor text
  addr_landmark  "near ..." / "opp ..." text
  addr_missing, addr_translit
"""
from __future__ import annotations

import re

import polars as pl
from anyascii import anyascii

from common import Paths, log, timed
from geo import explicit_state_rows

INDIC = r"[ऀ-෿]"
# Anusvara / tippi / bindu signs: nasal before a consonant, read as "n".
_ANUSVARA = re.compile("[ंঁংਂੰંଂஂంಂം]")

LEGAL = {
    "pvt": "private", "prvt": "private", "pvtltd": "private limited", "private": "private",
    "ltd": "limited", "limited": "limited", "ltda": "limited",
    "co": "company", "cos": "company", "company": "company",
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "opc": "opc",
    # French legal forms
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "ei": "ei",
    "eirl": "eirl", "sci": "sci", "snc": "snc", "selarl": "selarl", "scop": "scop",
}
# Consonant skeletons of legal words, for transliterated spellings ("praivet", "limtid").
LEGAL_SKEL = {"prvt": "private", "lmtd": "limited", "kmpn": "company"}
NAME_STOP = {"the", "of", "and", "a", "an", "le", "la", "les", "de", "du", "des", "et", "l", "d"}

ADDR_ABBR = {
    "US": {
        "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
        "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace",
        "trl": "trail", "sq": "square", "pt": "point", "mt": "mount", "ft": "fort",
        "expy": "expressway", "fwy": "freeway", "rte": "route",
        "n": "north", "s": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    },
    "India": {
        "rd": "road", "st": "street", "nr": "near", "opp": "opposite", "mkt": "market",
        "bldg": "building", "apt": "apartment", "apts": "apartments", "soc": "society",
        "sec": "sector", "indl": "industrial", "ind": "industrial", "est": "estate",
        "dist": "district", "distt": "district", "vill": "village", "vil": "village",
        "bengaluru": "bangalore", "gurugram": "gurgaon",
    },
    "France": {
        "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
        "blvd": "boulevard", "boul": "boulevard", "st": "saint", "ste": "sainte",
        "pl": "place", "all": "allee", "imp": "impasse", "ch": "chemin", "chem": "chemin",
        "rte": "route", "sq": "square", "fbg": "faubourg",
    },
}
# Safe for any country, including ones never seen in training.
ADDR_ABBR_GENERIC = {"rd": "road", "ave": "avenue", "blvd": "boulevard"}
ADDR_DROP = {"no", "nos", "number", "cdp", "city"}

HOUSE_RE = (r"(?:\b(?:plot|p|h|house|door|d|shop|kh|khasra|survey|sy|s|ward|building|bldg|"
            r"no|number|n)\s*(?:no|number)?\s*[.:#-]*\s*|#\s*)(\d+)")
UNIT_RE = (r"\b(?:unit|suite|ste|apt|apartment|flat|fl|floor|room|rm)\b\s*(?:no)?\s*[.:#-]*\s*"
           r"[0-9a-z/-]*|\b\d+\s*(?:st|nd|rd|th)?\s*(?:floor|fl)\b|\bground\s+floor\b")
LANDMARK_RE = r"^(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|adj)\b"


# --------------------------------------------------------------------------- helpers

def _to_ascii(text: str) -> str:
    return anyascii(_ANUSVARA.sub("n", text))


def fold_ascii(df: pl.DataFrame, col: str) -> pl.Expr:
    """Expression mapping `col` to ASCII; the slow Python call runs once per unique value."""
    uniq = df.filter(pl.col(col).str.contains(r"[^\x00-\x7F]"))[col].unique()
    if uniq.len() == 0:
        return pl.col(col)
    new = pl.Series([_to_ascii(s) for s in uniq.to_list()], dtype=pl.String)
    return pl.col(col).replace(uniq, new)


def skeleton(e: pl.Expr) -> pl.Expr:
    """Consonant skeleton: phonetic merges, vowels/h dropped after the first letter,
    doubled letters collapsed. 'praivet' and 'private' -> 'prvt'."""
    e = (e.str.replace_all("ph", "f", literal=True).str.replace_all("x", "ks", literal=True)
         .str.replace_all("gh", "", literal=True)
         .str.replace_all("th", "d", literal=True)  # English "th" is written as "d" in Indic scripts
         .str.replace_all(r"ti(on|al|a)", "sh${1}")
         .str.replace_all(r"c([eiy])", "s${1}")
         .str.replace_all("[cq]", "k").str.replace_all("w", "v", literal=True)
         .str.replace_all("z", "j", literal=True))
    e = e.str.slice(0, 1) + e.str.slice(1).str.replace_all("[aeiouyh]", "")
    for ch in "bcdfgjklmnprstv":
        e = e.str.replace_all(ch + ch, ch, literal=True)
    return e


def _num(e: pl.Expr) -> pl.Expr:
    """Strip leading zeros; an all-zero number becomes "0"."""
    e = e.str.strip_chars_start("0")
    return pl.when(e == "").then(pl.lit("0")).otherwise(e)


def _tokens(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


# --------------------------------------------------------------------------- names

def normalize_names(df: pl.DataFrame) -> pl.DataFrame:
    raw = pl.col("name")
    df = df.with_columns(
        raw.str.contains(INDIC).alias("name_translit"),
        fold_ascii(df, "name").str.to_lowercase().alias("_n"),
    )
    n = pl.col("_n")
    domain_re = r"(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|co\.in|in|co|biz|info|us|fr)\b"
    df = df.with_columns(
        n.str.extract(domain_re, 1).str.replace_all("-", "", literal=True).alias("name_domain")
    )
    n_clean = (
        n.str.replace_all(r"\|.*$", " ")                 # "| www.site.com" tails
        .str.replace_all(domain_re, " ")
        .str.replace_all(r"#\s*\d+", " ")                # "#99067"
        .str.replace_all("&", " and ", literal=True)
        .str.replace_all(r"[.']", "")                    # l.l.p. -> llp, pvt. -> pvt
        .str.replace_all(r"([a-z])0([a-z])", "${1}o${2}")  # OCR digits inside words
        .str.replace_all(r"([a-z])1([a-z])", "${1}l${2}")
        .str.replace_all(r"([a-z])3([a-z])", "${1}e${2}")
        .str.replace_all(r"([a-z])5([a-z])", "${1}s${2}")
        .str.replace_all(r"([a-z]{2})0\b", "${1}o")
        .str.replace_all(r"\b1lc\b", "llc")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
    )
    df = df.with_columns(_tokens(n_clean).alias("_tok"))
    # A name that is only a website becomes its domain label.
    df = df.with_columns(
        pl.when((pl.col("_tok").list.len() == 0) & pl.col("name_domain").is_not_null())
        .then(pl.concat_list(pl.col("name_domain")))
        .otherwise(pl.col("_tok")).alias("_tok")
    )
    # Token level: legal words (exact; plus skeleton match for transliterated names).
    t = pl.col("t")
    canon = pl.coalesce(
        t.replace_strict(LEGAL, default=None),
        pl.when(pl.col("name_translit") & (t.str.len_chars() >= 4))
        .then(skeleton(t).replace_strict(LEGAL_SKEL, default=None)),
    )
    tok = (df.select("rid", "name_translit", pl.col("_tok").alias("t"))
           .explode("t").filter(t.is_not_null())
           .with_columns(pl.int_range(pl.len()).over("rid").alias("pos"))
           .with_columns(canon.alias("canon")))
    is_core = pl.col("canon").is_null() & ~t.is_in(list(NAME_STOP))
    agg = tok.group_by("rid").agg(
        pl.coalesce("canon", "t").sort_by("pos").str.join(" ").alias("name_norm"),
        t.filter(is_core).sort_by(pl.col("pos").filter(is_core)).alias("_core"),
        pl.col("canon").drop_nulls().str.split(" ").flatten().unique().sort()
        .str.join(" ").alias("name_legal"),
    )
    df = df.join(agg, on="rid", how="left")
    return df.with_columns(
        pl.col("_core").list.sort().list.join(" ").alias("name_core"),
        pl.col("_core").list.join("").alias("name_concat"),
        pl.col("_core").list.eval(pl.element().str.slice(0, 1)).list.join("")
        .alias("name_initials"),
        pl.col("_core").list.eval(skeleton(pl.element())).list.sort().list.join(" ")
        .alias("name_skel"),
    ).drop("_n", "_tok", "_core")


# --------------------------------------------------------------------------- addresses

def _clean_comp(e: pl.Expr) -> pl.Expr:
    return (e.str.replace_all("&", " and ", literal=True)
            .str.replace_all(r"n°", " no ", literal=True)
            .str.replace_all(r"[.'`]", " ")
            .str.replace_all(r"[^a-z0-9/# -]+", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars(" -"))


def _abbr_expr(country: pl.Expr, tok: pl.Expr) -> pl.Expr:
    expr = tok.replace(ADDR_ABBR_GENERIC)
    for c, table in ADDR_ABBR.items():
        expr = pl.when(country == c).then(tok.replace(table)).otherwise(expr)
    return expr


def normalize_addresses(df: pl.DataFrame, aliases: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("address").is_null().alias("addr_missing"),
        pl.col("address").fill_null("").str.contains(INDIC).alias("addr_translit"),
        fold_ascii(df, "address").str.to_lowercase().alias("_a"),
    )
    comps = (
        df.select("rid", "country", pl.col("_a").str.split(",").alias("comp"))
        .explode("comp", empty_as_null=True)
        .filter(pl.col("comp").is_not_null())
        .with_columns(pl.int_range(pl.len()).over("rid").alias("pos"))
        .with_columns(_clean_comp(pl.col("comp")).alias("comp"))
        .filter(pl.col("comp") != "")
    )
    comps = comps.with_columns(
        pl.col("comp").str.contains(r"^(?:c/o|c o|care of)\b").alias("care_of"),
        pl.col("comp").str.contains(LANDMARK_RE).alias("landmark"),
        pl.col("comp").str.extract_all(UNIT_RE).list.join(" ").alias("unit"),
        pl.col("comp").str.replace_all("-", " ", literal=True).alias("key"),
    )
    explicit = pl.DataFrame(explicit_state_rows(), schema=["country", "key", "state"], orient="row")
    comps = (comps.join(explicit, on=["country", "key"], how="left")
             .join(aliases.rename({"state": "learned"}), on=["country", "key"], how="left"))
    # A learned alias that spells the state itself (e.g. transliterated "rajsthan") is a
    # state component; learned city names stay in the address text.
    def squashed_skel(c: str) -> pl.Expr:
        return skeleton(pl.col(c).str.replace_all(" ", "", literal=True))
    comps = comps.with_columns(
        pl.when(pl.col("state").is_null() & (squashed_skel("key") == squashed_skel("learned")))
        .then(pl.col("learned")).otherwise(pl.col("state")).alias("state")
    )
    keep = ~pl.col("care_of") & ~pl.col("landmark")
    comps = comps.with_columns(
        pl.col("comp").str.replace_all(UNIT_RE, " ").str.replace_all(r"\s+", " ")
        .str.strip_chars().alias("body")
    )
    comps = comps.with_columns(
        pl.col("body").str.extract_all(HOUSE_RE).list.eval(
            _num(pl.element().str.extract(r"(\d+)$", 1))).alias("house"),
        pl.col("body").str.extract_all(r"\d+").list.eval(_num(pl.element())).alias("nums"),
        pl.when(pl.col("body").str.contains("[a-z]"))
        .then(_num(pl.col("body").str.extract(r"^(\d+)[a-z]?\b", 1))).alias("lead"),
    )
    tok = (comps.filter(keep & pl.col("state").is_null())
           .select("rid", "country", "pos",
                   _tokens(pl.col("body").str.replace_all(r"[/#-]", " ")
                           .str.replace_all(r"\b(\d+)(?:st|nd|rd|th)\b", "${1}")).alias("t"))
           .explode("t").filter(pl.col("t").is_not_null())
           .with_columns(_abbr_expr(pl.col("country"), pl.col("t")).alias("t"))
           .filter(~pl.col("t").is_in(list(ADDR_DROP))))
    tok_agg = tok.group_by("rid").agg(
        pl.col("t").sort_by("pos").alias("_ordered"),
        pl.col("t").unique().sort().alias("_core"),
    )
    agg = comps.group_by("rid").agg(
        pl.col("state").filter(pl.col("state").is_not_null()).last().alias("_explicit"),
        pl.col("learned").filter(pl.col("learned").is_not_null()).mode().first().alias("_learned"),
        pl.col("house").filter(keep).flatten().drop_nulls().unique().alias("_house"),
        pl.col("lead").filter(keep).drop_nulls().first().alias("_lead"),
        pl.col("nums").filter(keep).flatten().drop_nulls().unique().sort().alias("addr_numbers"),
        pl.col("unit").filter(pl.col("unit") != "").str.join(" ").alias("addr_unit"),
        pl.col("comp").filter(pl.col("landmark")).str.join(" ").alias("addr_landmark"),
    )
    out = df.join(agg, on="rid", how="left").join(tok_agg, on="rid", how="left")
    return out.with_columns(
        pl.coalesce("_explicit", "_learned").alias("addr_state"),
        pl.col("_explicit").is_not_null().alias("addr_state_explicit"),
        pl.when(pl.col("_house").list.len() > 0).then(pl.col("_house"))
        .otherwise(pl.concat_list(pl.col("_lead")).list.drop_nulls()).alias("addr_house"),
        pl.col("_ordered").list.join(" ").alias("addr_norm"),
        pl.col("_core").list.join(" ").alias("addr_core"),
    ).drop("_a", "_explicit", "_learned", "_house", "_lead", "_ordered", "_core")


# --------------------------------------------------------------------------- learned states

def fit_state_aliases(records: pl.DataFrame, gt_pairs: pl.DataFrame) -> pl.DataFrame:
    """Learn address component -> state from matched S2/S3 records whose S1 names the
    state explicitly (native-script names, codes, cities). Training data only."""
    explicit = pl.DataFrame(explicit_state_rows(), schema=["country", "key", "state"], orient="row")
    df = records.select("rid", "entity_id", "source", "country", "address")
    df = df.with_columns(fold_ascii(df, "address").str.to_lowercase().alias("_a"))
    comps = (df.select("rid", "entity_id", "source", "country",
                       pl.col("_a").str.split(",").alias("comp"))
             .explode("comp", empty_as_null=True).filter(pl.col("comp").is_not_null())
             .with_columns(_clean_comp(pl.col("comp")).str.replace_all("-", " ", literal=True)
                           .alias("key"))
             .filter(pl.col("key") != "")
             .join(explicit, on=["country", "key"], how="left"))
    s1_state = (comps.filter((pl.col("source") == 1) & pl.col("state").is_not_null())
                .group_by("entity_id").agg(pl.col("state").last().alias("owner_state")))
    owner = gt_pairs.join(s1_state.rename({"entity_id": "s1_id"}), on="s1_id").select(
        pl.col("match_id").alias("entity_id"), "owner_state")
    learned = (comps.filter((pl.col("source") != 1) & pl.col("state").is_null()
                            & ~pl.col("key").str.contains(r"\d"))
               .join(owner, on="entity_id")
               .group_by("country", "key", "owner_state").len()
               .with_columns(pl.col("len").sum().over("country", "key").alias("total"))
               .filter((pl.col("total") >= 30) & (pl.col("len") / pl.col("total") >= 0.95))
               .select("country", "key", pl.col("owner_state").alias("state")))
    return learned


# --------------------------------------------------------------------------- stage

SUBSETS = {
    "sample": lambda p: [p.sample / "records.parquet", p.sample / "france.parquet"],
    "dev": lambda p: [p.slices / "dev_records.parquet"],
    "tune": lambda p: [p.slices / "tune_records.parquet"],
    "val": lambda p: [p.slices / "val_records.parquet"],
    "train": lambda p: [p.records("train")],
    "test": lambda p: [p.records("test")],
}
CHUNK_ROWS = 250_000  # ~1.5 GB peak per chunk; keeps 16 GB machines safe


def load_aliases(paths: Paths) -> pl.DataFrame:
    f = paths.norm / "state_aliases.parquet"
    if f.exists():
        return pl.read_parquet(f)
    with timed("fit learned state aliases (train minus val slice)"):
        records = pl.read_parquet(paths.records("train"))
        val_ids = pl.read_parquet(paths.slices / "val_records.parquet", columns=["rid"])
        records = records.join(val_ids, on="rid", how="anti")
        aliases = fit_state_aliases(records, pl.read_parquet(paths.gt_pairs()))
        aliases.write_parquet(f)
    log.info("learned %d state aliases", aliases.height)
    return aliases


def normalize(df: pl.DataFrame, aliases: pl.DataFrame) -> pl.DataFrame:
    df = normalize_names(df)
    df = normalize_addresses(df, aliases)
    return df.drop("name", "address")


def run(paths: Paths, subset: str) -> None:
    aliases = load_aliases(paths)
    df = pl.concat([pl.read_parquet(f) for f in SUBSETS[subset](paths)], how="diagonal")
    out_file = paths.norm / f"{subset}.parquet"
    with timed(f"normalize {subset} ({df.height} rows)"):
        parts = [normalize(df.slice(i, CHUNK_ROWS), aliases)
                 for i in range(0, df.height, CHUNK_ROWS)]
        out = pl.concat(parts)
        out.write_parquet(out_file)
    log.info("wrote %s | state found: %.1f%% (explicit %.1f%%) | house found: %.1f%%",
             out_file, 100 * out["addr_state"].is_not_null().mean(),
             100 * out["addr_state_explicit"].mean(),
             100 * (out["addr_house"].list.len() > 0).mean())
