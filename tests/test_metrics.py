from switchsafe.metrics import character_error_rate, word_error_rate


def test_word_error_rate_counts_substitution() -> None:
    result = word_error_rate("block my card", "block the card")
    assert result["wer"] == 1 / 3
    assert result["substitutions"] == 1


def test_character_error_rate_is_zero_after_normalization() -> None:
    assert character_error_rate("Hello, WORLD!", "hello world") == 0

