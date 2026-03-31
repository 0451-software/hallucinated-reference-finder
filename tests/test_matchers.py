"""Unit tests for title_matcher and author_matcher."""

from __future__ import annotations

import pytest

from halref.matching.author_matcher import (
    author_set_overlap,
    check_author_order,
    check_first_author,
    normalize_name,
    last_names_match,
)
from halref.matching.title_matcher import normalize_title, title_similarity
from halref.models import Author


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_authors(*last_names: str) -> list[Author]:
    return [Author(first="", last=name, full=name) for name in last_names]


# ---------------------------------------------------------------------------
# normalize_title
# ---------------------------------------------------------------------------

class TestNormalizeTitle:
    def test_empty_string_returns_empty(self):
        assert normalize_title("") == ""

    def test_lowercases(self):
        assert normalize_title("Hello World") == "hello world"

    def test_strips_punctuation(self):
        # Punctuation except hyphens should be removed
        result = normalize_title("Attention Is All You Need!")
        assert "!" not in result
        assert "attention is all you need" == result

    def test_preserves_hyphens(self):
        result = normalize_title("State-of-the-Art Approaches")
        assert "-" in result
        assert "state-of-the-art approaches" == result

    def test_collapses_whitespace(self):
        result = normalize_title("Too   Many    Spaces")
        assert "  " not in result
        assert result == "too many spaces"

    def test_unicode_normalization_nfkd(self):
        # NFKD decomposes accented chars; combining chars removed, lowercased
        assert normalize_title("Résumé") == "resume"

    def test_strips_trailing_whitespace(self):
        result = normalize_title("  padded  ")
        assert result == "padded"

    def test_colon_removed(self):
        # Colons are punctuation (non-word, non-hyphen) → removed
        result = normalize_title("BERT: Pre-training of Deep Models")
        assert ":" not in result

    @pytest.mark.parametrize("title,expected", [
        ("Hello World", "hello world"),
        ("  strip me  ", "strip me"),
        ("No,Punctuation!", "nopunctuation"),
        ("keep-hyphen", "keep-hyphen"),
    ])
    def test_parametrized_normalization(self, title: str, expected: str):
        assert normalize_title(title) == expected


# ---------------------------------------------------------------------------
# title_similarity
# ---------------------------------------------------------------------------

class TestTitleSimilarity:
    def test_identical_titles_return_one(self):
        assert title_similarity("Attention Is All You Need", "Attention Is All You Need") == 1.0

    def test_identical_after_normalization_return_one(self):
        # Differ only in case/punctuation
        assert title_similarity("Attention Is All You Need!", "attention is all you need") == 1.0

    def test_completely_different_titles_near_zero(self):
        # Titles with no shared tokens — token_set_ratio should still be low
        score = title_similarity(
            "Quantum Chromodynamics Lattice Field Theory Simulations",
            "Renaissance Baroque Architecture Cathedral Sculpture",
        )
        assert score < 0.4

    @pytest.mark.parametrize("a,b", [
        ("", "Some Title"),
        ("Some Title", ""),
        ("", ""),
    ])
    def test_empty_input_returns_zero(self, a, b):
        assert title_similarity(a, b) == 0.0

    def test_whitespace_differences_high_score(self):
        score = title_similarity("Neural Machine Translation", "Neural  Machine  Translation")
        assert score >= 0.95

    def test_minor_punctuation_difference_high_score(self):
        score = title_similarity("BERT: Pre-training of Deep Models", "BERT Pre-training of Deep Models")
        assert score >= 0.90

    def test_subtitle_variant_high_score(self):
        # Same main title, different subtitle
        score = title_similarity(
            "Language Models Are Few-Shot Learners",
            "Language Models Are Few Shot Learners",
        )
        assert score >= 0.90

    def test_partial_overlap_mid_range(self):
        score = title_similarity("Deep Neural Networks", "Deep Learning Networks")
        assert 0.3 < score < 1.0

    def test_score_in_unit_interval(self):
        for a, b in [
            ("foo", "bar"),
            ("abc", "abc"),
            ("completely", "different"),
        ]:
            s = title_similarity(a, b)
            assert 0.0 <= s <= 1.0, f"score {s} out of range for ({a!r}, {b!r})"


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_empty_returns_empty(self):
        assert normalize_name("") == ""

    def test_lowercases(self):
        assert normalize_name("Smith") == "smith"

    def test_strips_whitespace(self):
        assert normalize_name("  Jones  ") == "jones"

    def test_removes_accents(self):
        assert normalize_name("Müller") == "muller"


# ---------------------------------------------------------------------------
# last_names_match
# ---------------------------------------------------------------------------

class TestLastNamesMatch:
    def test_exact_match(self):
        assert last_names_match("Smith", "Smith") is True

    def test_case_insensitive_match(self):
        assert last_names_match("Smith", "smith") is True

    def test_empty_a_returns_false(self):
        assert last_names_match("", "Smith") is False

    def test_empty_b_returns_false(self):
        assert last_names_match("Smith", "") is False

    def test_completely_different_returns_false(self):
        assert last_names_match("Smith", "Johnson") is False

    def test_typo_within_threshold_matches(self):
        # "Smithe" vs "Smith" — very close
        assert last_names_match("Smithe", "Smith") is True

    def test_accent_variant(self):
        # Should match with fuzzy since ratio would be high
        result = last_names_match("Müller", "Muller")
        # After normalization both become "muller" (accent stripped)
        assert result is True


# ---------------------------------------------------------------------------
# author_set_overlap
# ---------------------------------------------------------------------------

class TestAuthorSetOverlap:
    def test_empty_authors_a_returns_zero(self):
        assert author_set_overlap([], make_authors("Smith")) == 0.0

    def test_empty_authors_b_returns_zero(self):
        assert author_set_overlap(make_authors("Smith"), []) == 0.0

    def test_both_empty_returns_zero(self):
        assert author_set_overlap([], []) == 0.0

    def test_identical_single_author_returns_one(self):
        a = make_authors("Smith")
        b = make_authors("Smith")
        assert author_set_overlap(a, b) == 1.0

    def test_identical_multi_author_returns_one(self):
        a = make_authors("Smith", "Jones", "Brown")
        b = make_authors("Smith", "Jones", "Brown")
        assert author_set_overlap(a, b) == 1.0

    def test_completely_different_returns_zero(self):
        a = make_authors("Smith", "Jones")
        b = make_authors("Brown", "White")
        assert author_set_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = make_authors("Smith", "Jones", "Brown")
        b = make_authors("Smith", "White", "Brown")
        score = author_set_overlap(a, b)
        # 2 match out of union of 4 (3+3-2)
        assert 0.0 < score < 1.0

    def test_single_vs_many_partial(self):
        a = make_authors("Smith")
        b = make_authors("Smith", "Jones", "Brown")
        score = author_set_overlap(a, b)
        # 1 match, union = 1+3-1=3, jaccard = 1/3
        assert abs(score - 1 / 3) < 0.01

    def test_score_in_unit_interval(self):
        a = make_authors("Smith", "Jones")
        b = make_authors("Brown", "Smith")
        score = author_set_overlap(a, b)
        assert 0.0 <= score <= 1.0

    def test_authors_with_no_last_name_handled(self):
        # Authors with empty last names should not crash
        a = [Author(first="John", last="", full="John")]
        b = make_authors("Smith")
        result = author_set_overlap(a, b)
        assert result == 0.0


# ---------------------------------------------------------------------------
# check_author_order
# ---------------------------------------------------------------------------

class TestCheckAuthorOrder:
    def test_empty_lists_returns_true(self):
        assert check_author_order([], []) is True

    def test_empty_a_returns_true(self):
        assert check_author_order([], make_authors("Smith")) is True

    def test_empty_b_returns_true(self):
        assert check_author_order(make_authors("Smith"), []) is True

    def test_single_author_returns_true(self):
        # Not enough pairs to check order
        a = make_authors("Smith")
        b = make_authors("Smith")
        assert check_author_order(a, b) is True

    def test_correct_order_two_authors(self):
        a = make_authors("Smith", "Jones")
        b = make_authors("Smith", "Jones")
        assert check_author_order(a, b) is True

    def test_reversed_order_returns_false(self):
        a = make_authors("Smith", "Jones")
        b = make_authors("Jones", "Smith")
        assert check_author_order(a, b) is False

    def test_three_authors_correct_order(self):
        a = make_authors("Smith", "Jones", "Brown")
        b = make_authors("Smith", "Jones", "Brown")
        assert check_author_order(a, b) is True

    def test_three_authors_reversed_returns_false(self):
        a = make_authors("Smith", "Jones", "Brown")
        b = make_authors("Brown", "Jones", "Smith")
        assert check_author_order(a, b) is False

    def test_partial_match_order_correct(self):
        # a has Smith+Jones; b has Smith+White+Jones — order still Smith before Jones
        a = make_authors("Smith", "Jones")
        b = make_authors("Smith", "White", "Jones")
        assert check_author_order(a, b) is True


# ---------------------------------------------------------------------------
# check_first_author
# ---------------------------------------------------------------------------

class TestCheckFirstAuthor:
    def test_same_first_author_returns_true(self):
        a = make_authors("Smith", "Jones")
        b = make_authors("Smith", "Brown")
        assert check_first_author(a, b) is True

    def test_different_first_author_returns_false(self):
        a = make_authors("Smith", "Jones")
        b = make_authors("Brown", "Jones")
        assert check_first_author(a, b) is False

    def test_empty_a_returns_true(self):
        assert check_first_author([], make_authors("Smith")) is True

    def test_empty_b_returns_true(self):
        assert check_first_author(make_authors("Smith"), []) is True
