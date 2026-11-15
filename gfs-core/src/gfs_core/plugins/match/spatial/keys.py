from __future__ import annotations

import re

KNOWN_FORMATIONS = {
    "433", "442", "4231", "352", "532", "541", "4222", "343",
    "4141", "451", "3421",
}


def normalize_formation_key(formation: str) -> str:
    if not formation:
        return "433"
    primary = str(formation).split(" - ")[0].split("–")[0].strip()
    numbers = re.findall(r"\d+", primary.replace(" ", ""))
    if not numbers:
        return "433"
    key = "".join(numbers[:4])
    if key in KNOWN_FORMATIONS:
        return key
    if len(numbers) >= 3:
        return "".join(numbers[:3])
    return "433"
