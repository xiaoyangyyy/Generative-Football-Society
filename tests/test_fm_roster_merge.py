from src.data_engine.fm_roster_merge import build_fm_only_roster


def test_build_fm_only_roster_produces_complete_provenance():
    players = [
        {
            'name': f'Player {index}',
            'ca': 170 - index,
            'age': 21 + index % 8,
            'club': 'Test Club',
            'position': 'GK' if index == 0 else 'CM',
        }
        for index in range(12)
    ]

    roster = build_fm_only_roster('Test Team', players, formation='4-3-3')

    assert roster['source'] == 'fm_export'
    assert roster['squad_size'] == 12
    assert len(roster['players']) == 12
    assert len({player['player_id'] for player in roster['players']}) == 12
    assert all(player['source'] == 'fm_export' for player in roster['players'])
    assert 'team_dynamics' in roster
