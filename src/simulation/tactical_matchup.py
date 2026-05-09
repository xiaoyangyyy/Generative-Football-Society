STYLE_MATCHUP_MATRIX = {
    ("high_press", "possession_control"): 3.0,
    ("high_press", "low_block_counter"): -2.0,
    ("high_press", "direct_vertical"): 1.0,
    ("possession_control", "low_block_counter"): 2.5,
    ("possession_control", "direct_vertical"): -1.5,
    ("low_block_counter", "direct_vertical"): 2.0,
}


def compute_matchup_bonus(style_a, style_b):
    if style_a == style_b:
        return 0.0
    if (style_a, style_b) in STYLE_MATCHUP_MATRIX:
        return STYLE_MATCHUP_MATRIX[(style_a, style_b)]
    if (style_b, style_a) in STYLE_MATCHUP_MATRIX:
        return -STYLE_MATCHUP_MATRIX[(style_b, style_a)]
    return 0.0
