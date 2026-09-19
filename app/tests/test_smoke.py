def test_state_imports():
    from state import MainQueryState

    s = MainQueryState()
    assert s.messages == []
    assert s.query_type is None
