from code_search.ingest.tokenizer import count_tokens


def test_count_tokens_scales_with_code_length():
    short = count_tokens("def foo(): pass")
    long = count_tokens("def foo(self, a, b=2):\n    return a + b\n" * 20)
    assert 0 < short < long


def test_count_tokens_counts_indentation():
    assert count_tokens("x = 1") < count_tokens("        x = 1")
