TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "UEFA Euro": 0.85,
    "Copa América": 0.85,
    "African Cup of Nations": 0.70,
    "AFC Asian Cup": 0.70,
    "Gold Cup": 0.70,
    "OFC Nations Cup": 0.60,
    "FIFA World Cup qualification": 0.60,
    "UEFA Euro qualification": 0.55,
    "African Cup of Nations qualification": 0.45,
    "AFC Asian Cup qualification": 0.45,
    "CONCACAF Nations League": 0.45,
    "UEFA Nations League": 0.50,
    "Friendly": 0.20
}

DEFAULT_WEIGHT = 0.35

def get_tournament_weight(tournament_name: str) -> float:
    return TOURNAMENT_WEIGHTS.get(tournament_name, DEFAULT_WEIGHT)
