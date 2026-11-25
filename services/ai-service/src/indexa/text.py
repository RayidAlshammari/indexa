"""Arabic/English text normalisation and tokenisation for evidence retrieval.

Government evidence is mostly Arabic PDFs while the NDMO specifications are
published in English, so retrieval works on normalised tokens from both
languages. Normalisation folds the spelling variants that make the same Arabic
word look different to a lexical index (alef forms, taa marbuta, diacritics,
tatweel) and strips presentation-form glyphs some PDF producers emit.
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)  # escaped: combining marks reorder when copied
_TATWEEL = "\u0640"
_TOKEN = re.compile(r"[ء-ي٠-٩]+|[a-z0-9]+")

_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ات", "ون", "ين", "ها", "هم", "ية", "ه", "ة")

STOPWORDS = frozenset(
    """
    في من على الى إلى عن مع او أو ثم ان أن إن كان كانت هذه هذا ذلك التي الذي الذين
    كل اي أي عند بين حيث قد لا ما لم لن هو هي هم يتم يجب تم وفق وفقا ضمن خلال
    the of and to in for on by with a an or be is are shall this that as at from
    its any all each such which within entity entitys including include
    """.split()
)


def normalize(text: str) -> str:
    """Fold presentation forms, diacritics and common Arabic spelling variants."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    # Many Word-exported Arabic PDFs store the lam-alef ligature as alef+lam
    # ("الإسلامية" extracts as "اإلسالمية"). Fold both orders to one key so the
    # same word matches whichever way the producer encoded it.
    text = text.replace("لا", "ال")
    return text.lower()


def _stem_ar(token: str) -> str:
    for p in _AR_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p) :]
            break
    for s in _AR_SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token.replace("ة", "ه")


def tokenize(text: str) -> list[str]:
    """Normalised, light-stemmed tokens with stopwords removed."""
    out = []
    for tok in _TOKEN.findall(normalize(text)):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if "ء" <= tok[0] <= "ي":
            tok = _stem_ar(tok)
        elif len(tok) > 4 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


def squash(text: str) -> str:
    """Normalised text with all whitespace and punctuation removed, for quote matching."""
    return re.sub(r"[^\w]", "", normalize(text))
