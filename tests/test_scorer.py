"""Unit tests for scorer.py — composite hallucination scoring."""

from __future__ import annotations

import pytest

from halref.config import MatchingWeights
from halref.matching.scorer import score_reference, _select_best_match
from halref.models import APIMatch, APISource, Author, MatchResult, Reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ref(
    title: str = "Some Reference Title",
    authors: list[Author] | None = None,
    year: int | None = 2022,
    doi: str = "",
) -> Reference:
    return Reference(
        raw_text="raw",
        title=title,
        authors=authors or [],
        year=year,
        doi=doi,
    )


def make_match(
    title: str = "Some Reference Title",
    authors: list[Author] | None = None,
    year: int | None = 2022,
    doi: str = "",
) -> APIMatch:
    return APIMatch(
        source=APISource.SEMANTIC_SCHOLAR,
        title=title,
        authors=authors or [],
        year=year,
        doi=doi,
    )


def make_author(last: str, first: str = "") -> Author:
    return Author(first=first, last=last, full=f"{first} {last}".strip())


# ---------------------------------------------------------------------------
# No API matches
# ---------------------------------------------------------------------------

class TestNoAPIMatches:
    def test_no_matches_returns_high_score(self):
        ref = make_ref()
        result = score_reference(ref, [])
        assert result.hallucination_score > 0.9

    def test_no_matches_has_no_match_signal(self):
        ref = make_ref()
        result = score_reference(ref, [])
        names = [s.name for s in result.signals]
        assert "no_match" in names

    def test_no_matches_result_has_no_best_match(self):
        ref = make_ref()
        result = score_reference(ref, [])
        assert result.best_match is None


# ---------------------------------------------------------------------------
# DOI override
# ---------------------------------------------------------------------------

class TestDOIOverride:
    def test_doi_match_forces_low_score(self):
        doi = "10.1234/test.doi"
        ref = make_ref(doi=doi)
        match = make_match(doi=doi)
        result = score_reference(ref, [match])
        assert result.hallucination_score < 0.1

    def test_doi_match_adds_doi_match_signal(self):
        doi = "10.1234/test.doi"
        ref = make_ref(doi=doi)
        match = make_match(doi=doi)
        result = score_reference(ref, [match])
        names = [s.name for s in result.signals]
        assert "doi_match" in names

    def test_doi_mismatch_no_override(self):
        ref = make_ref(doi="10.1234/ref-doi")
        match = make_match(doi="10.9999/other-doi")
        result = score_reference(ref, [match])
        # Should NOT return 0.05; score is computed normally
        assert result.hallucination_score != 0.05

    def test_missing_doi_no_override(self):
        ref = make_ref(doi="")
        match = make_match(doi="10.1234/something")
        result = score_reference(ref, [match])
        # doi_matches returns None when one side is missing, no override
        names = [s.name for s in result.signals]
        assert "doi_match" not in names

    def test_doi_case_insensitive(self):
        ref = make_ref(doi="10.1234/TEST.DOI")
        match = make_match(doi="10.1234/test.doi")
        result = score_reference(ref, [match])
        assert result.hallucination_score == 0.05


# ---------------------------------------------------------------------------
# Title-only match signal
# ---------------------------------------------------------------------------

class TestTitleSignal:
    def test_identical_title_low_title_mismatch_signal(self):
        title = "Attention Is All You Need"
        ref = make_ref(title=title)
        match = make_match(title=title)
        result = score_reference(ref, [match])
        title_signal = next(s for s in result.signals if s.name == "title_mismatch")
        assert title_signal.value < 0.1

    def test_different_title_high_title_mismatch_signal(self):
        ref = make_ref(title="Attention Is All You Need")
        match = make_match(title="Supply Chain Optimization Techniques")
        result = score_reference(ref, [match])
        title_signal = next(s for s in result.signals if s.name == "title_mismatch")
        assert title_signal.value > 0.5

    def test_title_similarity_stored_in_result(self):
        title = "Attention Is All You Need"
        ref = make_ref(title=title)
        match = make_match(title=title)
        result = score_reference(ref, [match])
        assert result.title_similarity >= 0.99


# ---------------------------------------------------------------------------
# Author signal
# ---------------------------------------------------------------------------

class TestAuthorSignal:
    def test_matching_authors_low_mismatch(self):
        authors = [make_author("Vaswani"), make_author("Shazeer")]
        ref = make_ref(authors=authors)
        match = make_match(authors=authors)
        result = score_reference(ref, [match])
        author_signal = next(s for s in result.signals if s.name == "author_mismatch")
        assert author_signal.value < 0.1

    def test_different_authors_high_mismatch(self):
        ref_authors = [make_author("Smith"), make_author("Jones")]
        match_authors = [make_author("Brown"), make_author("White")]
        ref = make_ref(authors=ref_authors)
        match = make_match(authors=match_authors)
        result = score_reference(ref, [match])
        author_signal = next(s for s in result.signals if s.name == "author_mismatch")
        assert author_signal.value > 0.5

    def test_author_overlap_stored_in_result(self):
        authors = [make_author("Vaswani")]
        ref = make_ref(authors=authors)
        match = make_match(authors=authors)
        result = score_reference(ref, [match])
        assert result.author_overlap == 1.0


# ---------------------------------------------------------------------------
# Combined signals
# ---------------------------------------------------------------------------

class TestCombinedSignals:
    def test_perfect_match_very_low_score(self):
        title = "BERT: Pre-training of Deep Bidirectional Transformers"
        authors = [make_author("Devlin"), make_author("Chang"), make_author("Lee")]
        ref = make_ref(title=title, authors=authors, year=2019)
        match = make_match(title=title, authors=authors, year=2019)
        result = score_reference(ref, [match])
        assert result.hallucination_score < 0.3

    def test_completely_wrong_match_high_score(self):
        ref = make_ref(
            title="Deep Learning for NLP",
            authors=[make_author("Smith")],
            year=2020,
        )
        match = make_match(
            title="Supply Chain Logistics and Warehousing",
            authors=[make_author("Johnson"), make_author("Brown")],
            year=2010,
        )
        result = score_reference(ref, [match])
        assert result.hallucination_score > 0.4

    def test_signals_all_present(self):
        ref = make_ref()
        match = make_match()
        result = score_reference(ref, [match])
        expected_signals = {"title_mismatch", "author_mismatch", "author_order_wrong",
                            "year_mismatch", "low_api_consensus"}
        actual_signals = {s.name for s in result.signals}
        assert expected_signals.issubset(actual_signals)

    def test_score_in_unit_interval(self):
        ref = make_ref(title="Some Paper", authors=[make_author("Smith")], year=2020)
        match = make_match(title="Different Paper", authors=[make_author("Jones")], year=2015)
        result = score_reference(ref, [match])
        assert 0.0 <= result.hallucination_score <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_reference_no_crash(self):
        ref = Reference(raw_text="", title="", authors=[], year=None)
        match = make_match()
        result = score_reference(ref, [match])
        assert isinstance(result, MatchResult)

    def test_missing_year_no_crash(self):
        ref = make_ref(year=None)
        match = make_match(year=None)
        result = score_reference(ref, [match])
        assert isinstance(result, MatchResult)

    def test_no_authors_no_crash(self):
        ref = make_ref(authors=[])
        match = make_match(authors=[])
        result = score_reference(ref, [match])
        assert isinstance(result, MatchResult)

    def test_multiple_matches_picks_best(self):
        title = "Attention Is All You Need"
        ref = make_ref(title=title)
        good_match = make_match(title=title)
        bad_match = make_match(title="Completely Irrelevant Supply Chain Paper")
        result = score_reference(ref, [bad_match, good_match])
        # Best match should be the good one
        assert result.title_similarity >= 0.99

    def test_custom_weights_accepted(self):
        weights = MatchingWeights(title=0.5, authors=0.5, author_order=0.0, year=0.0, consensus=0.0)
        ref = make_ref()
        match = make_match()
        result = score_reference(ref, [match], weights=weights)
        assert 0.0 <= result.hallucination_score <= 1.0

    def test_year_mismatch_increases_score(self):
        ref = make_ref(title="Same Title", year=2010)
        match = make_match(title="Same Title", year=2020)  # 10 years off
        result = score_reference(ref, [match])
        year_signal = next(s for s in result.signals if s.name == "year_mismatch")
        assert year_signal.value > 0.0

    def test_correct_year_zero_mismatch_signal(self):
        ref = make_ref(title="Same Title", year=2022)
        match = make_match(title="Same Title", year=2022)
        result = score_reference(ref, [match])
        year_signal = next(s for s in result.signals if s.name == "year_mismatch")
        assert year_signal.value == 0.0


# ---------------------------------------------------------------------------
# _select_best_match (internal helper)
# ---------------------------------------------------------------------------

class TestSelectBestMatch:
    def test_selects_best_by_title_similarity(self):
        ref = make_ref(title="Attention Is All You Need")
        matches = [
            make_match(title="Completely Different Topic"),
            make_match(title="Attention Is All You Need"),
        ]
        best, sim = _select_best_match(ref, matches)
        assert best.title == "Attention Is All You Need"
        assert sim >= 0.99

    def test_selects_better_of_two_matches(self):
        ref = make_ref(title="Attention Is All You Need")
        worse = make_match(title="Some Unrelated Paper About Supply Chains")
        better = make_match(title="Attention Is All You Need")
        best, sim = _select_best_match(ref, [worse, better])
        assert best is better
        assert sim >= 0.99
