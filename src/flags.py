"""Flagi reprezentacji: nazwa drużyny (konwencja results.csv) -> kod ISO 3166-1 alpha-2.

Kody używane są na dwa sposoby:

* ``flag_url`` - obraz PNG z flagcdn.com (renderuje się wszędzie, także w Windows, gdzie
  przeglądarki nie mają glifów flag-emoji),
* ``flag_emoji`` - emoji (do kontrolek tekstowych, np. list rozwijanych).

Reprezentacje brytyjskie używają kodów subdywizji (``gb-eng``, ``gb-sct``, ``gb-wls``, ``gb-nir``).
"""
from __future__ import annotations

from typing import Dict

ISO_CODES: Dict[str, str] = {
    # AFC
    "Afghanistan": "af", "Australia": "au", "Bahrain": "bh", "Bangladesh": "bd",
    "Bhutan": "bt", "Brunei": "bn", "Cambodia": "kh", "China": "cn", "Guam": "gu",
    "Hong Kong": "hk", "India": "in", "Indonesia": "id", "Iran": "ir", "Iraq": "iq",
    "Japan": "jp", "Jordan": "jo", "Kuwait": "kw", "Kyrgyzstan": "kg", "Laos": "la",
    "Lebanon": "lb", "Macau": "mo", "Malaysia": "my", "Maldives": "mv", "Mongolia": "mn",
    "Myanmar": "mm", "Nepal": "np", "North Korea": "kp", "Oman": "om", "Pakistan": "pk",
    "Palestine": "ps", "Philippines": "ph", "Qatar": "qa", "Saudi Arabia": "sa",
    "Singapore": "sg", "South Korea": "kr", "Sri Lanka": "lk", "Syria": "sy",
    "Taiwan": "tw", "Tajikistan": "tj", "Thailand": "th", "Timor-Leste": "tl",
    "Turkmenistan": "tm", "United Arab Emirates": "ae", "Uzbekistan": "uz",
    "Vietnam": "vn", "Yemen": "ye",
    # CAF
    "Algeria": "dz", "Angola": "ao", "Benin": "bj", "Botswana": "bw", "Burkina Faso": "bf",
    "Burundi": "bi", "Cameroon": "cm", "Cape Verde": "cv", "Central African Republic": "cf",
    "Chad": "td", "Comoros": "km", "Congo": "cg", "DR Congo": "cd", "Djibouti": "dj",
    "Egypt": "eg", "Equatorial Guinea": "gq", "Eritrea": "er", "Eswatini": "sz",
    "Ethiopia": "et", "Gabon": "ga", "Gambia": "gm", "Ghana": "gh", "Guinea": "gn",
    "Guinea-Bissau": "gw", "Ivory Coast": "ci", "Kenya": "ke", "Lesotho": "ls",
    "Liberia": "lr", "Libya": "ly", "Madagascar": "mg", "Malawi": "mw", "Mali": "ml",
    "Mauritania": "mr", "Mauritius": "mu", "Morocco": "ma", "Mozambique": "mz",
    "Namibia": "na", "Niger": "ne", "Nigeria": "ng", "Rwanda": "rw",
    "São Tomé and Príncipe": "st", "Senegal": "sn", "Seychelles": "sc",
    "Sierra Leone": "sl", "Somalia": "so", "South Africa": "za", "South Sudan": "ss",
    "Sudan": "sd", "Tanzania": "tz", "Togo": "tg", "Tunisia": "tn", "Uganda": "ug",
    "Zambia": "zm", "Zimbabwe": "zw",
    # CONCACAF
    "Anguilla": "ai", "Antigua and Barbuda": "ag", "Aruba": "aw", "Bahamas": "bs",
    "Barbados": "bb", "Belize": "bz", "Bermuda": "bm", "British Virgin Islands": "vg",
    "Canada": "ca", "Cayman Islands": "ky", "Costa Rica": "cr", "Cuba": "cu",
    "Curaçao": "cw", "Dominica": "dm", "Dominican Republic": "do", "El Salvador": "sv",
    "Grenada": "gd", "Guatemala": "gt", "Guyana": "gy", "Haiti": "ht", "Honduras": "hn",
    "Jamaica": "jm", "Mexico": "mx", "Montserrat": "ms", "Nicaragua": "ni", "Panama": "pa",
    "Puerto Rico": "pr", "Saint Kitts and Nevis": "kn", "Saint Lucia": "lc",
    "Saint Vincent and the Grenadines": "vc", "Suriname": "sr", "Trinidad and Tobago": "tt",
    "Turks and Caicos Islands": "tc", "United States": "us",
    "United States Virgin Islands": "vi",
    # CONMEBOL
    "Argentina": "ar", "Bolivia": "bo", "Brazil": "br", "Chile": "cl", "Colombia": "co",
    "Ecuador": "ec", "Paraguay": "py", "Peru": "pe", "Uruguay": "uy", "Venezuela": "ve",
    # OFC
    "American Samoa": "as", "Cook Islands": "ck", "Fiji": "fj", "New Caledonia": "nc",
    "New Zealand": "nz", "Papua New Guinea": "pg", "Samoa": "ws", "Solomon Islands": "sb",
    "Tahiti": "pf", "Tonga": "to", "Vanuatu": "vu",
    # UEFA
    "Albania": "al", "Andorra": "ad", "Armenia": "am", "Austria": "at", "Azerbaijan": "az",
    "Belarus": "by", "Belgium": "be", "Bosnia and Herzegovina": "ba", "Bulgaria": "bg",
    "Croatia": "hr", "Cyprus": "cy", "Czech Republic": "cz", "Denmark": "dk",
    "England": "gb-eng", "Estonia": "ee", "Faroe Islands": "fo", "Finland": "fi",
    "France": "fr", "Georgia": "ge", "Germany": "de", "Gibraltar": "gi", "Greece": "gr",
    "Hungary": "hu", "Iceland": "is", "Israel": "il", "Italy": "it", "Kazakhstan": "kz",
    "Kosovo": "xk", "Latvia": "lv", "Liechtenstein": "li", "Lithuania": "lt",
    "Luxembourg": "lu", "Malta": "mt", "Moldova": "md", "Montenegro": "me",
    "Netherlands": "nl", "North Macedonia": "mk", "Northern Ireland": "gb-nir",
    "Norway": "no", "Poland": "pl", "Portugal": "pt", "Republic of Ireland": "ie",
    "Romania": "ro", "Russia": "ru", "San Marino": "sm", "Scotland": "gb-sct",
    "Serbia": "rs", "Slovakia": "sk", "Slovenia": "si", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "Turkey": "tr", "Ukraine": "ua", "Wales": "gb-wls",
}


def iso_code(team: str) -> str:
    """Kod ISO drużyny (małymi literami) albo pusty string, gdy brak w słowniku."""
    return ISO_CODES.get(team, "")


def flag_url(team: str, width: int = 40) -> str:
    """Adres obrazka flagi (PNG, flagcdn.com); pusty string dla nieznanej drużyny."""
    code = iso_code(team)
    return f"https://flagcdn.com/w{width}/{code}.png" if code else ""


def flag_emoji(team: str) -> str:
    """Flaga jako emoji (regionalne wskaźniki lub sekwencja tagów dla Anglii/Szkocji/Walii)."""
    code = iso_code(team)
    if not code:
        return ""
    if "-" in code:  # subdywizje UK: czarna flaga + znaki tagów + znacznik końca
        tags = "".join(chr(0xE0000 + ord(c)) for c in code.replace("-", ""))
        return "\U0001F3F4" + tags + "\U000E007F"
    if code == "xk":
        return "\U0001F1FD\U0001F1F0"
    return "".join(chr(0x1F1E6 + ord(c) - ord("a")) for c in code)
