from fluid_motion.core.gpu import flicker_risk


def test_rtx_50_series_flags_risky():
    assert flicker_risk("NVIDIA GeForce RTX 5070 Ti") is True
    assert flicker_risk("NVIDIA GeForce RTX 5090") is True
    assert flicker_risk("NVIDIA GeForce RTX 5070 Ti Laptop GPU") is True


def test_older_rtx_series_are_not_risky():
    assert flicker_risk("NVIDIA GeForce RTX 4090") is False
    assert flicker_risk("NVIDIA GeForce RTX 3080") is False
    assert flicker_risk("NVIDIA GeForce RTX 2080 Ti") is False


def test_unknown_or_unparseable_name_defaults_to_risky():
    assert flicker_risk("") is True
    assert flicker_risk("NVIDIA RTX PRO 6000 Blackwell") is True
    assert flicker_risk("Some Unknown Card") is True
