from src.simulation.llm_engine import SimulationLLM


def build_parser_only_llm():
    # Bypass __init__ because this smoke test only validates local JSON extraction.
    return SimulationLLM.__new__(SimulationLLM)


def main():
    llm = build_parser_only_llm()

    cases = [
        ('{"winner":"A","score":"1-0"}', {"winner": "A", "score": "1-0"}),
        (
            "```json\n"
            '{"winner":"B","score":"2-1"}\n'
            "```",
            {"winner": "B", "score": "2-1"},
        ),
        (
            'model output: analysis first {"winner":"C","score":"0-0"} trailing text',
            {"winner": "C", "score": "0-0"},
        ),
    ]

    for raw, expected in cases:
        parsed = llm._extract_json_object(raw)
        assert parsed == expected, f"Expected {expected}, got {parsed}"

    bad_cases = ["", "no json object here", "```json\n[1,2,3]\n```"]
    for raw in bad_cases:
        try:
            llm._extract_json_object(raw)
            raise AssertionError(f"Expected parser to fail for: {raw!r}")
        except RuntimeError:
            pass

    print("smoke_llm_json_parser: PASS")


if __name__ == "__main__":
    main()
