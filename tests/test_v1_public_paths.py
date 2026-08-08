def test_v1_public_modules_import() -> None:
    import backend.replay_generate  # noqa: F401
    import backend.shadow_audit  # noqa: F401
    import backend.simulation.runtime  # noqa: F401
    import backend.simulation.shadow  # noqa: F401
    import backend.simulation.store  # noqa: F401
    import backend.providers.openai_compatible  # noqa: F401
