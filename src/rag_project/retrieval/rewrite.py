"""Conservative query normalisation.

Two rules govern this module:

1. Expansions are *additive*. "TB" becomes "TB tuberculosis", never
   "tuberculosis". Replacing the abbreviation would destroy the exact lexical
   token that BM25 needs, trading a gain in the dense half for a loss in the
   lexical half.

2. The glossary is curated by hand and deliberately small. An aggressive or
   model-generated expansion is how a query about one condition quietly starts
   retrieving another -- "MI" is myocardial infarction here, but it is also
   mitral incompetence, and guessing wrong in a medical corpus is not a
   recoverable error. Ambiguous abbreviations are left alone on purpose.
"""

from __future__ import annotations

import re

# Curated for the MOHFW Standard Treatment Guidelines domain. Entries are
# expanded only where the expansion is unambiguous in this corpus.
GLOSSARY: dict[str, str] = {
    "tb": "tuberculosis",
    "mdr-tb": "multidrug resistant tuberculosis",
    "xdr-tb": "extensively drug resistant tuberculosis",
    "dots": "directly observed treatment short course",
    "naat": "nucleic acid amplification test",
    "cbnaat": "cartridge based nucleic acid amplification test",
    "ors": "oral rehydration solution",
    "anc": "antenatal care",
    "pnc": "postnatal care",
    "pph": "postpartum haemorrhage",
    "gdm": "gestational diabetes mellitus",
    "ncd": "non communicable disease",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "uti": "urinary tract infection",
    "urti": "upper respiratory tract infection",
    "lrti": "lower respiratory tract infection",
    "ari": "acute respiratory infection",
    "art": "antiretroviral therapy",
    "hiv": "human immunodeficiency virus",
    "sti": "sexually transmitted infection",
    "rti": "reproductive tract infection",
    "sam": "severe acute malnutrition",
    "mam": "moderate acute malnutrition",
    "ifa": "iron and folic acid",
    "bcg": "bacillus calmette guerin",
    "opv": "oral polio vaccine",
    "aes": "acute encephalitis syndrome",
    "je": "japanese encephalitis",
    "imnci": "integrated management of neonatal and childhood illness",
    "phc": "primary health centre",
    "chc": "community health centre",
    "asha": "accredited social health activist",
    "anm": "auxiliary nurse midwife",
    "t2dm": "type 2 diabetes mellitus",
    "htn": "hypertension",
}

# Deliberately NOT expanded -- each has more than one reading in this corpus.
AMBIGUOUS = frozenset({"mi", "ms", "as", "ra", "cp", "pid", "dm", "bp", "hb", "ca"})

_WORD = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def expand_abbreviations(query: str) -> tuple[str, list[str]]:
    """Append expansions for known abbreviations. Returns (query, applied)."""
    applied: list[str] = []
    seen: set[str] = set()
    for token in _WORD.findall(query):
        low = token.lower()
        if low in AMBIGUOUS or low in seen:
            continue
        expansion = GLOSSARY.get(low)
        if expansion and expansion.lower() not in query.lower():
            applied.append(f"{token} -> {expansion}")
            seen.add(low)
    if not applied:
        return query.strip(), []
    extra = " ".join(GLOSSARY[a.split(" -> ")[0].lower()] for a in applied)
    return f"{query.strip()} {extra}", applied


def normalise(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def rewrite(query: str) -> tuple[str, list[str]]:
    """Full rewrite pass. The log of what changed is returned for the trace --
    a silent rewrite is untraceable when an answer later looks wrong."""
    return expand_abbreviations(normalise(query))
