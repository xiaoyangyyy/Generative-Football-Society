"""Parse group stage results from full_run_latest.log."""
import itertools
import re
from pathlib import Path

GROUPS = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "Group B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "Group C": ["Brazil", "Scotland", "Morocco", "Haiti"],
    "Group D": ["United States", "Turkey", "Australia", "Paraguay"],
    "Group E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "Group F": ["Netherlands", "Sweden", "Japan", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "Group I": ["France", "Norway", "Senegal", "Iraq"],
    "Group J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "Group K": ["Portugal", "Colombia", "DR Congo", "Uzbekistan"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"],
}

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "full_run_latest.log"


def main():
    text = LOG.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"\[SCORE\]\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s+\|")
    parsed = [
        (m.group(1).strip(), int(m.group(2)), int(m.group(3)), m.group(4).strip())
        for m in pat.finditer(text)
    ]
    gs = parsed[:72]

    expected = []
    for g, teams in GROUPS.items():
        for t1, t2 in itertools.combinations(teams, 2):
            expected.append((g, t1, t2))

    standings = {
        g: {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0} for t in teams}
        for g, teams in GROUPS.items()
    }
    by_group = {g: [] for g in GROUPS}
    total_goals = 0

    for (g, t1, t2), (p1, s1, s2, p2) in zip(expected, gs):
        if p1 != t1 or p2 != t2:
            print(f"WARN mismatch {g}: expected {t1} vs {t2}, got {p1} {s1}-{s2} {p2}")
        total_goals += s1 + s2
        by_group[g].append((t1, s1, s2, t2))
        for t, gf, ga in ((t1, s1, s2), (t2, s2, s1)):
            st = standings[g][t]
            st["gf"] += gf
            st["ga"] += ga
            st["gd"] = st["gf"] - st["ga"]
            if gf > ga:
                st["pts"] += 3
                st["w"] += 1
            elif gf == ga:
                st["pts"] += 1
                st["d"] += 1
            else:
                st["l"] += 1

    print("=== GROUP STAGE SUMMARY ===")
    print(f"Matches: {len(gs)}  Total goals: {total_goals}  Avg: {total_goals / len(gs):.2f}/match\n")

    advancers = []
    for g in GROUPS:
        tbl = sorted(
            standings[g].items(),
            key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]),
        )
        print(f"--- {g} ---")
        for i, (t, st) in enumerate(tbl, 1):
            mark = " → R32" if i <= 2 else ""
            print(
                f"  {i}. {t:28} {st['pts']}pts  "
                f"{st['w']}W{st['d']}D{st['l']}L  {st['gf']}-{st['ga']} ({st['gd']:+d}){mark}"
            )
            if i <= 2:
                advancers.append(t)
        results = ", ".join(f"{a} {s1}-{s2} {b}" for a, s1, s2, b in by_group[g])
        print(f"  Results: {results}\n")

    high = sorted(gs, key=lambda x: x[1] + x[2], reverse=True)[:10]
    low = [x for x in gs if x[1] + x[2] == 0]
    print("=== HIGHEST SCORING MATCHES ===")
    for p1, s1, s2, p2 in high:
        print(f"  {p1} {s1}-{s2} {p2}  ({s1 + s2} goals)")

    print(f"\n=== 0-0 DRAWS: {len(low)} ===")
    for p1, s1, s2, p2 in low:
        print(f"  {p1} {s1}-{s2} {p2}")

    all_teams = {t: st for g in GROUPS for t, st in standings[g].items()}
    print("\n=== TOP ATTACK (goals scored) ===")
    for t, st in sorted(all_teams.items(), key=lambda x: -x[1]["gf"])[:12]:
        print(f"  {t}: {st['gf']} GF, {st['ga']} GA, {st['pts']} pts")

    print("\n=== WORST DEFENCE (goals conceded) ===")
    for t, st in sorted(all_teams.items(), key=lambda x: -x[1]["ga"])[:8]:
        print(f"  {t}: {st['ga']} conceded in 3 matches")

    print("\n=== ELIMINATED (0 pts) ===")
    for g in GROUPS:
        for t, st in standings[g].items():
            if st["pts"] == 0:
                print(f"  {g}: {t} ({st['gf']}-{st['ga']})")

    print("\n=== QUALIFIERS (24 teams) ===")
    print(", ".join(advancers))


if __name__ == "__main__":
    main()
