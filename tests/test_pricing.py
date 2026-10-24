"""Tests for centralized pricing configuration and cost calculations."""

from unittest.mock import patch

from catcher_in_the_rag.pricing import (
    calculate_cost_usd,
    get_model_rates,
    load_pricing_config,
)


def test_load_pricing_config_from_file() -> None:
    config = load_pricing_config()
    assert "models" in config
    assert "gemini-3.5-flash" in config["models"]
    assert config["models"]["gemini-3.5-flash"]["prompt_cost_per_1m"] == 0.50
    assert config["models"]["gemini-3.5-flash"]["candidate_cost_per_1m"] == 3.00


def test_get_model_rates_known_and_fallback() -> None:
    p_rate, c_rate = get_model_rates("gemini-3.5-flash")
    assert p_rate == 0.50
    assert c_rate == 3.00

    # Prefix match with provider path
    p_rate, c_rate = get_model_rates("vertex_ai/gemini-3.5-flash")
    assert p_rate == 0.50
    assert c_rate == 3.00

    # Legacy rate
    p_leg, c_leg = get_model_rates("gemini-1.5-flash-legacy")
    assert p_leg == 0.075
    assert c_leg == 0.30

    # Unknown model fallback to gemini-3.5-flash
    p_unk, c_unk = get_model_rates("unknown-future-model")
    assert p_unk == 0.50
    assert c_unk == 3.00


def test_calculate_cost_usd() -> None:
    # 1,000,000 prompt tokens @ $0.50 + 1,000,000 candidate tokens @ $3.00 = $3.50
    cost = calculate_cost_usd(1_000_000, 1_000_000, "gemini-3.5-flash")
    assert abs(cost - 3.50) < 1e-6

    # 10,000 prompt tokens + 1,000 candidate tokens
    cost_small = calculate_cost_usd(10_000, 1_000, "gemini-3.5-flash")
    expected = (10_000 * 0.50 / 1e6) + (1_000 * 3.00 / 1e6)
    assert abs(cost_small - expected) < 1e-6


def test_load_pricing_fallback_on_missing_file() -> None:
    with patch("catcher_in_the_rag.pricing._PRICING_CONFIG_PATH") as mock_path:
        mock_path.exists.return_value = False
        config = load_pricing_config()
        assert config["source"] == "fallback"
        assert "gemini-3.5-flash" in config["models"]
