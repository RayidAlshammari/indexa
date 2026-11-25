"""Parse the gold 'evidence' field ("pdp p12-p14, p18; notice p37") into doc+page keys."""

import re


def gold_pages(evidence: str) -> set[str]:
    keys, doc = set(), None
    for part in re.split(r"[;,]", evidence or ""):
        part = part.strip()
        m = re.match(r"(pdp|dc|notice)\s+(.*)", part)
        if m:
            doc, part = m.group(1), m.group(2)
        for a, b in re.findall(r"p(\d+)(?:-p(\d+))?", part):
            for p in range(int(a), int(b or a) + 1):
                keys.add(f"{doc}{p}")
    return keys
