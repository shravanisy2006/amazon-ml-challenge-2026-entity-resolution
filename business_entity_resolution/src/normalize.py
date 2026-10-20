"""
normalize.py — record cleaning shared by every later stage.

Design rules
  * Country is an open set of strings. Nothing here filters on {US, India}.
    Country-specific tables (state names) are only *lookups*; unknown
    countries (e.g. France in test) fall through to generic cleaning.
  * No external data or services. All tables below are hand-written.
  * Every function is pure (str -> str/set) so it can run in multiprocessing.
"""
import re
import unicodedata

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    HAVE_TRANSLIT = True
except ImportError:          # pipeline still runs, just weaker on Indic names
    HAVE_TRANSLIT = False

# ───────────────────────── script detection ─────────────────────────
_SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),   ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),      ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),     ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
]

def detect_script(text: str) -> str:
    """Dominant Indic script in text, or 'latin' if none."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        if cp < 0x0900:
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "latin"

# ───────────────────────── Indic → Latin ─────────────────────────
_SCHEMES = {}
if HAVE_TRANSLIT:
    _SCHEMES = {
        "devanagari": sanscript.DEVANAGARI, "bengali": sanscript.BENGALI,
        "gurmukhi": sanscript.GURMUKHI, "gujarati": sanscript.GUJARATI,
        "oriya": sanscript.ORIYA, "tamil": sanscript.TAMIL,
        "telugu": sanscript.TELUGU, "kannada": sanscript.KANNADA,
        "malayalam": sanscript.MALAYALAM,
    }
_INDIC_RUN = re.compile(r"[\u0900-\u0D7F\u200c\u200d]+")

def _romanize_run(run: str) -> str:
    script = detect_script(run)
    if script not in _SCHEMES:
        return run
    out = _translit(run, _SCHEMES[script], sanscript.IAST)
    out = out.replace("ṃ", "n").replace("ṁ", "n")
    # drop the inherent short final 'a' (rāma -> rām, limiṭeḍa -> limiṭeḍ);
    # long ā is kept (mīḍiyā)
    out = re.sub(r"(?<=[^aeiouāīūṛ\s])a\b", "", out)
    return out

# legal-form words in Indic scripts, mapped directly (transliteration mangles them)
_INDIC_LEGAL = {
    "प्रा. लि.": " private limited ", "प्राइवेट": " private ", "लिमिटेड": " limited ",
    "एलएलपी": " llp ", "எல்எல்பி": " llp ", "பிரைவேட்": " private ", "லிமிடெட்": " limited ",
    "ಪ್ರೈವೇಟ್": " private ", "ಲಿಮಿಟೆಡ್": " limited ", "ప్రైవేట్": " private ",
    "లిమిటెడ్": " limited ", "প্রাইভেট": " private ", "লিমিটেড": " limited ",
    "પ્રાઇવેટ": " private ", "લિમિટેડ": " limited ", "പ്രൈവറ്റ്": " private ",
    "ലിമിറ്റഡ്": " limited ", "ପ୍ରାଇଭେଟ୍": " private ", "ଲିମିଟେଡ୍": " limited ",
}

def romanize(text: str) -> str:
    """Replace every Indic-script run with its approximate Latin spelling."""
    if not _INDIC_RUN.search(text):
        return text
    for native, latin in _INDIC_LEGAL.items():
        text = text.replace(native, latin)
    text = text.replace("ॉ", "ो").replace("ॅ", "े")   # English 'o'/'e' loan vowels
    if not HAVE_TRANSLIT or not _INDIC_RUN.search(text):
        return text
    return _INDIC_RUN.sub(lambda m: " " + _romanize_run(m.group()) + " ", text)

# ───────────────────────── basic helpers ─────────────────────────
def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")

def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

_NULL_TOKENS = re.compile(r"(<null>|\bnull\b|\bn/a\b|\bnone\b|\bnan\b)", re.I)

def is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or str(x).strip() == ""

# ───────────────────────── names ─────────────────────────
# alias markers: "Nexnovi DBA Aguilar Diversified LLC" -> primary + alias
_ALIAS_SPLIT = re.compile(
    r"\s+(?:d\.?\s?b\.?\s?a\.?:?|d/b/a|doing business as|t/a|trading as|"
    r"a\.?k\.?a\.?|a/k/a|f\.?k\.?a\.?|f/k/a|formerly known as|formerly:?)\s+",
    re.I)
# decorations that carry no identity
_NAME_JUNK = [
    re.compile(r"\(id:\s*\d+\)", re.I),          # (ID: 93523)
    re.compile(r"#\d{3,}"),                       # #80430
    re.compile(r"[-–]\s*\d{7,}"),                 # - 7306204978
    re.compile(r"\|\s*www\.\S+", re.I),           # | www.x.com
    re.compile(r"^[\s\-\.>*@#]+"),                # leading -- >> ... *** @ #
]
_HONORIFICS = re.compile(r"^(m/s|messrs|shri|sri|smt|dr|mr|mrs|ms|the)\.?\s+", re.I)
_URL = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-\.]+?)\.?(com|net|org|in|co\.in|fr|io|biz)$", re.I)
_URL_NODOT = re.compile(r"^([a-z]{6,})(com)$")   # 'daoncontinentalmusiccom'
_DOTTED = re.compile(r"\b[a-z](?:\.\s?[a-z])+\.?(?=\W|$)")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "6": "g", "8": "b"})

# legal forms: US, India, and generic European/French (France is test-only)
LEGAL_WORDS = {
    "llc", "inc", "incorporated", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "llp", "lp", "pllc", "pc", "plc", "co", "company", "gmbh",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "selarl", "scop", "cie",
    "ag", "bv", "nv", "spa", "srl", "public",
}
_LEGAL_EXPAND = {"pvt": "private", "ltd": "limited", "corp": "corporation",
                 "inc": "incorporated", "co": "company"}
# generic business words that pad names ("Summit Health LLC Center")
FILLER_WORDS = {"center", "centre", "services", "service", "partners", "group",
                "enterprises", "and", "of", "the", "et", "de", "du", "des", "la",
                "le", "les", "l", "d"}

def _fix_leet(token: str) -> str:
    """'8lankenship'->'blankenship', 'G1obal'->'global'; leaves '13', 'c5' alone."""
    letters = sum(c.isalpha() for c in token)
    if letters >= 2 and any(c.isdigit() for c in token):
        return token.translate(_LEET)
    return token

def clean_name(raw) -> dict:
    """
    Returns
      norm     cleaned full name (lowercase, romanized, no accents/punct)
      core     norm minus legal forms and filler words
      compact  core with spaces removed (compares against URL-style names)
      alias    secondary name from a DBA/AKA/FKA pattern, cleaned the same way
      is_url   whether the name was a domain / handle
      script   dominant script of the raw name
    """
    if is_blank(raw):
        return dict(norm="", core="", compact="", alias="", is_url=False, script="latin")
    text = unicodedata.normalize("NFKC", str(raw)).strip()
    script = detect_script(text)
    text = romanize(text)

    parts = _ALIAS_SPLIT.split(text, maxsplit=1)
    primary, alias = parts[0], (parts[1] if len(parts) > 1 else "")

    def _one(t):
        handle = t.strip()[:1] in "@#" and " " not in t.strip()
        for pat in _NAME_JUNK:
            t = pat.sub(" ", t)
        t = _HONORIFICS.sub("", t.strip())
        t = strip_accents(t).lower()
        is_url = False
        compact_try = t.strip().lstrip("@#")
        m = _URL.match(compact_try) or _URL_NODOT.match(compact_try)
        if (m or t.strip()[:1] in "@#") and " " not in compact_try:
            if not m:
                m = re.match(r"(.*)", compact_try)
            t, is_url = m.group(1).replace(".", " ").replace("-", " "), True
        t = _DOTTED.sub(lambda m: m.group().replace(".", "").replace(" ", ""), t)
        t = t.replace("&", " and ").replace("+", " and ")
        t = re.sub(r"[^\w\s]", " ", t)
        toks = [_fix_leet(w) for w in t.split()]
        toks = [_LEGAL_EXPAND.get(w, w) for w in toks]
        dedup = [w for i, w in enumerate(toks) if i == 0 or w != toks[i - 1]]
        return " ".join(dedup), (is_url or handle)

    norm, is_url = _one(primary)
    alias_norm = _one(alias)[0] if alias else ""
    core_toks = [w for w in norm.split()
                 if w not in LEGAL_WORDS and w not in FILLER_WORDS
                 and _LEGAL_EXPAND.get(w, w) not in LEGAL_WORDS]
    core = " ".join(core_toks) or norm
    return dict(norm=norm, core=core, compact=core.replace(" ", ""),
                alias=alias_norm, is_url=is_url, script=script)

# ───────────────────────── addresses ─────────────────────────
_ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "trl": "trail", "ter": "terrace", "cv": "cove",
    "sq": "square", "nr": "near", "opp": "opposite", "apt": "apartment",
    "fl": "floor", "flr": "floor", "bldg": "building", "no": "number",
    "h": "house", "hn": "house", "saint": "street",   # 'FREMONT SAINT' noise
    "twp": "township", "twnship": "township",
}
_ADDR_ABBR_BY_COUNTRY = {
    "france": {"r": "rue", "av": "avenue", "bd": "boulevard", "bld": "boulevard",
               "imp": "impasse", "all": "allee", "ch": "chemin", "che": "chemin",
               "rte": "route", "fbg": "faubourg", "pl": "place", "sq": "square",
               "qu": "quai", "crs": "cours", "pass": "passage", "res": "residence"},
}
_ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4",
                  "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
                  "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
                  "thirteenth": "13", "fourteenth": "14", "fifteenth": "15"}
_ADDR_JUNK = re.compile(r"\b(pmb|po box|p\.o\. box)\s*\d+\b", re.I)

# state tables are country-scoped lookups; unknown countries skip them
US_STATES = {
 "al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california",
 "co":"colorado","ct":"connecticut","de":"delaware","dc":"district of columbia",
 "fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois",
 "in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana",
 "me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan",
 "mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana",
 "ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey",
 "nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota",
 "oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania",
 "ri":"rhode island","sc":"south carolina","sd":"south dakota",
 "tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia",
 "wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming"}
IN_STATES = {
 "mh":"maharashtra","ka":"karnataka","tn":"tamil nadu","wb":"west bengal",
 "gj":"gujarat","hr":"haryana","br":"bihar","up":"uttar pradesh",
 "mp":"madhya pradesh","rj":"rajasthan","dl":"delhi","tg":"telangana",
 "ts":"telangana","kl":"kerala","ap":"andhra pradesh","od":"odisha",
 "or":"odisha","pb":"punjab","jh":"jharkhand","cg":"chhattisgarh",
 "uk":"uttarakhand","hp":"himachal pradesh","as":"assam","ga":"goa",
 "jk":"jammu and kashmir","ch":"chandigarh","py":"puducherry",
 "orissa":"odisha","keralam":"kerala"}
# native-script state names, matched *before* romanization
_IN_NATIVE = {"महाराष्ट्र":"maharashtra","ಕರ್ನಾಟಕ":"karnataka","தமிழ்நாடு":"tamil nadu",
 "পশ্চিমবঙ্গ":"west bengal","ગુજરાત":"gujarat","हरियाणा":"haryana","बिहार":"bihar",
 "उत्तर प्रदेश":"uttar pradesh","मध्य प्रदेश":"madhya pradesh","राजस्थान":"rajasthan",
 "दिल्ली":"delhi","తెలంగాణ":"telangana","കേരളം":"kerala","ଓଡ଼ିଶା":"odisha",
 "ఆంధ్ర ప్రదేశ్":"andhra pradesh","ਪੰਜਾਬ":"punjab"}
_STATE_TABLES = {"us": US_STATES, "india": IN_STATES}

def _norm_number(tok: str):
    """'005208'->'5208', '1056c'->'1056', '45th'/'45nd'->'45'. None if not numeric-led."""
    m = re.match(r"^0*(\d+)[a-z]{0,2}$", tok)
    return m.group(1) if m else None

def clean_address(raw, country: str) -> dict:
    """
    Returns
      norm    cleaned address text (abbreviations expanded, state canonical)
      tokens  alpha tokens (for Jaccard / blocking)
      nums    set of normalized numeric tokens
      state   canonical state if recognised, else ''
    """
    if is_blank(raw):
        return dict(norm="", tokens=set(), nums=set(), state="", labels=[])
    text = unicodedata.normalize("NFKC", str(raw))
    for native, canon in _IN_NATIVE.items():
        text = text.replace(native, canon)
    text = romanize(text)
    text = _NULL_TOKENS.sub(" ", text)
    text = _ADDR_JUNK.sub(" ", text)
    text = strip_accents(text).lower()
    text = re.sub(r"\b\d+\s*/\s*2\b", " ", text)           # '1/2'
    text = re.sub(r"[^\w\s,/\-]", " ", text)

    table = _STATE_TABLES.get(str(country).strip().lower(), {})
    local_abbr = _ADDR_ABBR_BY_COUNTRY.get(str(country).strip().lower(), {})
    full_names = set(table.values())
    out_parts, nums, abbr_hits, full_hits, labels = [], set(), [], [], []
    for comp in text.split(","):
        comp = re.sub(r"[/\-]", " ", comp)
        comp_toks = []
        for tok in comp.split():
            tok = _ORDINAL_WORDS.get(tok, tok)
            n = _norm_number(tok)
            if n is not None:
                nums.add(n); comp_toks.append(n)
            else:
                comp_toks.append(tok)
        comp_str = " ".join(comp_toks)
        if comp_toks and not any(c.isdigit() for c in comp_str):
            lt = ["saint" if (i == 0 and t in ("st", "ste")) else t for i, t in enumerate(comp_toks)]
            labels.append(" ".join(lt))
        # a component that is *entirely* a state abbreviation or state name
        if comp_str in table:
            comp_str = table[comp_str]; abbr_hits.append(comp_str)
        elif comp_str in full_names:
            full_hits.append(comp_str)
        else:
            comp_str = " ".join(local_abbr.get(t, _ADDR_ABBR.get(t, t)) for t in comp_toks)
        out_parts.append(comp_str)
    norm = _squash(" ".join(out_parts))
    # "Washington, DC" -> DC (abbreviation wins); "District of Columbia, .., Washington"
    # -> first full name wins
    state = abbr_hits[-1] if abbr_hits else (full_hits[0] if full_hits else "")
    tokens = {t for t in norm.split() if not t.isdigit() and len(t) > 1}
    return dict(norm=norm, tokens=tokens, nums=nums, state=state, labels=labels)

def clean_country(raw) -> str:
    return "" if is_blank(raw) else _squash(strip_accents(str(raw)).lower())


# ───────────────────────── sound skeleton ─────────────────────────
# Transliterated Indic names sound like, but are not spelled like, their English
# originals ("istarn inphra" vs "eastern infra"). A consonant skeleton maps
# both to the same key: merge sound-alike letters, drop vowels, collapse
# doubles. "eastern infra" and "istarn inphra" both become "strn nfr".
_SKEL_RULES = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"),
               ("dh", "d"), ("sh", "s"), ("ch", "c"), ("jh", "j"), ("ck", "k"),
               ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v"), ("c", "k")]

def sound_skeleton(name: str) -> str:
    out = []
    for w in name.split():
        for a, b in _SKEL_RULES:
            w = w.replace(a, b)
        w = re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy]", "", w))
        if w:
            out.append(w)
    return " ".join(out)
