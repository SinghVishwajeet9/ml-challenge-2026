"""Static geographic vocabularies (domain knowledge, no external lookups)."""

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC", "PR", "GU", "VI", "AS", "MP",
}

# Canonical Indian state / UT name -> accepted spellings (lowercase).
INDIA_STATES = {
    "andhra pradesh": ["andhra pradesh"],
    "arunachal pradesh": ["arunachal pradesh"],
    "assam": ["assam"],
    "bihar": ["bihar"],
    "chhattisgarh": ["chhattisgarh", "chattisgarh"],
    "goa": ["goa"],
    "gujarat": ["gujarat"],
    "haryana": ["haryana"],
    "himachal pradesh": ["himachal pradesh"],
    "jharkhand": ["jharkhand"],
    "karnataka": ["karnataka"],
    "kerala": ["kerala"],
    "madhya pradesh": ["madhya pradesh"],
    "maharashtra": ["maharashtra"],
    "manipur": ["manipur"],
    "meghalaya": ["meghalaya"],
    "mizoram": ["mizoram"],
    "nagaland": ["nagaland"],
    "odisha": ["odisha", "orissa"],
    "punjab": ["punjab"],
    "rajasthan": ["rajasthan"],
    "sikkim": ["sikkim"],
    "tamil nadu": ["tamil nadu", "tamilnadu"],
    "telangana": ["telangana"],
    "tripura": ["tripura"],
    "uttar pradesh": ["uttar pradesh"],
    "uttarakhand": ["uttarakhand", "uttaranchal"],
    "west bengal": ["west bengal"],
    "andaman and nicobar islands": ["andaman and nicobar islands", "andaman & nicobar islands"],
    "chandigarh": ["chandigarh"],
    "dadra and nagar haveli and daman and diu": [
        "dadra and nagar haveli and daman and diu", "dadra and nagar haveli",
        "dadra & nagar haveli", "daman and diu", "daman & diu",
    ],
    "delhi": ["delhi", "new delhi", "nct of delhi"],
    "jammu and kashmir": ["jammu and kashmir", "jammu & kashmir"],
    "ladakh": ["ladakh"],
    "lakshadweep": ["lakshadweep"],
    "puducherry": ["puducherry", "pondicherry"],
}

INDIA_ALIASES = {alias: canon for canon, aliases in INDIA_STATES.items() for alias in aliases}

US_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "district of columbia": "DC", "puerto rico": "PR",
}

INDIA_STATE_CODES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "or": "odisha", "od": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
    "an": "andaman and nicobar islands", "ch": "chandigarh",
    "dn": "dadra and nagar haveli and daman and diu",
    "dd": "dadra and nagar haveli and daman and diu", "dl": "delhi",
    "jk": "jammu and kashmir", "la": "ladakh", "ld": "lakshadweep", "py": "puducherry",
}

# France: metropolitan regions, and departments mapped to their region.
# Keys are ASCII, lowercase, hyphens replaced by spaces.
FRANCE_REGIONS = {
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
    "corse", "grand est", "hauts de france", "ile de france", "normandie",
    "nouvelle aquitaine", "occitanie", "pays de la loire", "provence alpes cote d azur",
}
FRANCE_DEPARTMENTS = {
    "nord": "hauts de france", "pas de calais": "hauts de france", "aisne": "hauts de france",
    "oise": "hauts de france", "somme": "hauts de france",
    "gironde": "nouvelle aquitaine", "landes": "nouvelle aquitaine",
    "pyrenees atlantiques": "nouvelle aquitaine", "lot et garonne": "nouvelle aquitaine",
    "dordogne": "nouvelle aquitaine", "charente": "nouvelle aquitaine",
    "charente maritime": "nouvelle aquitaine", "correze": "nouvelle aquitaine",
    "creuse": "nouvelle aquitaine", "deux sevres": "nouvelle aquitaine",
    "vienne": "nouvelle aquitaine", "haute vienne": "nouvelle aquitaine",
    "loire atlantique": "pays de la loire", "maine et loire": "pays de la loire",
    "mayenne": "pays de la loire", "sarthe": "pays de la loire", "vendee": "pays de la loire",
    "paris": "ile de france", "seine saint denis": "ile de france",
    "hauts de seine": "ile de france", "val de marne": "ile de france",
    "rhone": "auvergne rhone alpes", "bouches du rhone": "provence alpes cote d azur",
    "haute garonne": "occitanie", "bas rhin": "grand est", "ille et vilaine": "bretagne",
}


def explicit_state_rows() -> list[tuple[str, str, str]]:
    """(country, component, state) rows; components are normalized ASCII lowercase."""
    rows = [("US", c.lower(), c) for c in US_STATES]
    rows += [("US", name, code) for name, code in US_STATE_NAMES.items()]
    rows += [("India", a, s) for a, s in INDIA_ALIASES.items()]
    rows += [("India", a, s) for a, s in INDIA_STATE_CODES.items()]
    rows += [("France", r, r) for r in FRANCE_REGIONS]
    rows += [("France", d, r) for d, r in FRANCE_DEPARTMENTS.items()]
    return rows
