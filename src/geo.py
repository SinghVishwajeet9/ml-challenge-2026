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
