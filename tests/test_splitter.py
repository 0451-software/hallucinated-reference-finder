"""Unit tests for extract/splitter.py — reference block splitting."""

from __future__ import annotations

import pytest

from halref.extract.splitter import (
    split_references,
    dehyphenate,
    _split_by_blank_lines,
    _split_by_numbers,
    _split_by_author_year_pattern,
    _merge_fragments,
)


# ---------------------------------------------------------------------------
# split_references — high-level
# ---------------------------------------------------------------------------

class TestSplitReferences:
    def test_empty_input_returns_empty_list(self):
        assert split_references("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_references("   \n\n  ") == []

    def test_single_reference_returned_as_one(self):
        text = "Vaswani, A., Shazeer, N., and Parmar, N. (2017). Attention is all you need. NeurIPS."
        result = split_references(text)
        assert len(result) >= 1
        # The single reference should be returned
        combined = " ".join(result)
        assert "Vaswani" in combined

    def test_blank_line_separated_references(self):
        text = (
            "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS 2017.\n"
            "\n"
            "Devlin, J., Chang, M., Lee, K. (2019). BERT: Pre-training transformers. NAACL 2019.\n"
            "\n"
            "Brown, T., Mann, B., Ryder, N. (2020). Language models are few-shot learners. NeurIPS 2020.\n"
        )
        result = split_references(text)
        assert len(result) >= 2

    def test_numbered_references_square_brackets(self):
        text = (
            "[1] Vaswani, A. et al. (2017). Attention is all you need. In NeurIPS 2017, pages 5998–6008.\n"
            "[2] Devlin, J. et al. (2019). BERT: Pre-training of transformers. In NAACL 2019.\n"
            "[3] Brown, T. et al. (2020). Language models are few-shot learners. NeurIPS 2020.\n"
        )
        result = split_references(text)
        assert len(result) == 3

    def test_author_year_format(self):
        text = (
            "Tom Brown, Benjamin Mann, Nick Ryder, et al. 2020. "
            "Language Models are Few-Shot Learners. NeurIPS.\n"
            "Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. "
            "BERT: Pre-training of Deep Bidirectional Transformers. NAACL.\n"
            "Yinhan Liu, Myle Ott, Naman Goyal, et al. 2019. "
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv.\n"
        )
        result = split_references(text)
        assert len(result) >= 2

    def test_returns_list_of_strings(self):
        text = (
            "[1] First reference. Author, A. 2020. Title one. Journal A.\n"
            "[2] Second reference. Author, B. 2021. Title two. Journal B.\n"
        )
        result = split_references(text)
        assert all(isinstance(r, str) for r in result)
        assert all(len(r) > 0 for r in result)

    def test_malformed_input_no_crash(self):
        text = "This is just some random text without any reference markers at all."
        result = split_references(text)
        # Should not crash; may return the whole text or empty
        assert isinstance(result, list)

    def test_references_do_not_contain_only_whitespace(self):
        text = (
            "Brown, T., Mann, B. (2020). Language models. NeurIPS 2020.\n"
            "\n"
            "Devlin, J., Chang, M. (2019). BERT. NAACL 2019.\n"
        )
        result = split_references(text)
        assert all(r.strip() for r in result)


# ---------------------------------------------------------------------------
# _split_by_blank_lines
# ---------------------------------------------------------------------------

class TestSplitByBlankLines:
    def test_two_references_separated_by_blank_line(self):
        text = (
            "Author, A. (2020). Title one. Journal A.\n"
            "\n"
            "Author, B. (2021). Title two. Journal B.\n"
        )
        result = _split_by_blank_lines(text)
        assert len(result) == 2

    def test_three_references(self):
        text = (
            "Author, A. (2020). Title one. Journal.\n"
            "\n"
            "Author, B. (2021). Title two. Conference.\n"
            "\n"
            "Author, C. (2022). Title three. Workshop.\n"
        )
        result = _split_by_blank_lines(text)
        assert len(result) == 3

    def test_skips_very_short_chunks(self):
        text = "A real reference from 2020 with full details here.\n\nok\n\nAnother real reference 2021 full details."
        result = _split_by_blank_lines(text)
        # "ok" is < 20 chars and should be dropped
        assert all(len(r) >= 20 for r in result)

    def test_joins_wrapped_lines_within_reference(self):
        text = (
            "Author, A. (2020). A long title that wraps\n"
            "to the next line. Journal of Things.\n"
            "\n"
            "Author, B. (2021). Another reference. Conference.\n"
        )
        result = _split_by_blank_lines(text)
        # Wrapped lines should be joined with space
        assert "\n" not in result[0]

    def test_empty_input_returns_empty(self):
        assert _split_by_blank_lines("") == []


# ---------------------------------------------------------------------------
# _split_by_numbers
# ---------------------------------------------------------------------------

class TestSplitByNumbers:
    def test_three_numbered_refs(self):
        text = (
            "[1] Vaswani et al. (2017). Attention is all you need. NeurIPS 2017.\n"
            "[2] Devlin et al. (2019). BERT. NAACL 2019.\n"
            "[3] Brown et al. (2020). GPT-3. NeurIPS 2020.\n"
        )
        result = _split_by_numbers(text)
        assert len(result) == 3

    def test_content_after_marker_preserved(self):
        text = "[1] First author et al. 2020. Title of Paper. Conference.\n[2] Second author. 2021. Other Title. Journal."
        result = _split_by_numbers(text)
        assert "First author" in result[0]
        assert "Second author" in result[1]

    def test_single_number_returns_empty(self):
        text = "[1] Only one reference. Author, A. (2020). Some title. Journal."
        result = _split_by_numbers(text)
        # Only one marker — can't split
        assert result == []

    def test_empty_input_returns_empty(self):
        assert _split_by_numbers("") == []

    def test_very_short_chunks_filtered(self):
        text = "[1] A\n[2] This is a proper reference. Author, B. 2021. Title. Journal.\n[3] Another proper reference. Author, C. 2022. Title. Venue."
        result = _split_by_numbers(text)
        # "[1] A" chunk is < 20 chars after stripping, should be filtered
        for ref in result:
            assert len(ref) > 20


# ---------------------------------------------------------------------------
# _split_by_author_year_pattern
# ---------------------------------------------------------------------------

class TestSplitByAuthorYearPattern:
    def test_multiple_author_year_refs(self):
        text = (
            "Tom Brown, Benjamin Mann, Nick Ryder. 2020. Language Models Are Few-Shot Learners. NeurIPS.\n"
            "Jacob Devlin, Ming-Wei Chang, Kenton Lee. 2019. BERT. NAACL.\n"
            "Yinhan Liu, Myle Ott, Naman Goyal. 2019. RoBERTa. arXiv preprint.\n"
        )
        result = _split_by_author_year_pattern(text)
        assert len(result) >= 2

    def test_empty_input_returns_empty(self):
        assert _split_by_author_year_pattern("") == []

    def test_blocks_contain_year(self):
        text = (
            "Tom Brown, Benjamin Mann. 2020. Language Models. NeurIPS.\n"
            "Jacob Devlin, Ming-Wei Chang. 2019. BERT. NAACL.\n"
        )
        result = _split_by_author_year_pattern(text)
        import re
        for ref in result:
            assert re.search(r"\b(?:19|20)\d{2}\b", ref), f"No year in: {ref!r}"


# ---------------------------------------------------------------------------
# dehyphenate
# ---------------------------------------------------------------------------

class TestDehyphenate:
    def test_removes_hyphen_at_line_break(self):
        text = "represen-\ntations"
        result = dehyphenate(text)
        assert "representations" in result
        assert "-\n" not in result

    def test_removes_hyphen_at_line_break_with_spaces(self):
        text = "represen- \n  tations"
        result = dehyphenate(text)
        assert "representations" in result

    def test_removes_inline_hyphen_space(self):
        text = "represen- tations"
        result = dehyphenate(text)
        assert "representations" in result

    def test_preserves_legitimate_hyphens(self):
        # Hyphens in compound words on same line (no space) should be preserved
        text = "state-of-the-art approach"
        result = dehyphenate(text)
        assert "state-of-the-art" in result

    def test_empty_string_unchanged(self):
        assert dehyphenate("") == ""

    def test_no_hyphens_unchanged(self):
        text = "no hyphens here at all"
        assert dehyphenate(text) == text


# ---------------------------------------------------------------------------
# _merge_fragments
# ---------------------------------------------------------------------------

class TestMergeFragments:
    def test_empty_list_returns_empty(self):
        assert _merge_fragments([]) == []

    def test_single_item_returned_as_is(self):
        refs = ["A standalone reference. Author, 2020."]
        assert _merge_fragments(refs) == refs

    def test_merges_lowercase_continuation(self):
        refs = [
            "Author, A. and Author, B. 2020. Title of paper. In",
            "Proceedings of the Conference on Things, pages 1–10.",
        ]
        # Second starts with "Proceedings" — _FRAGMENT_START should match
        result = _merge_fragments(refs)
        assert len(result) == 1
        assert "Proceedings" in result[0]

    def test_does_not_merge_independent_references(self):
        refs = [
            "Author Smith, Jones. 2020. Real Paper Title. Journal of Things, 10(2):100-120.",
            "Author Brown, White. 2021. Another Paper. Conference Proceedings 2021, pages 50-60.",
        ]
        result = _merge_fragments(refs)
        assert len(result) == 2

    def test_merges_pagination_fragment(self):
        refs = [
            "Vaswani, A. et al. 2017. Attention is all you need. NeurIPS,",
            "pages 5998–6008.",
        ]
        result = _merge_fragments(refs)
        assert len(result) == 1
        assert "pages" in result[0]

    @pytest.mark.parametrize("fragment_start,should_merge", [
        ("volume 10, pages 1-10.", True),
        ("pages 100-200.", True),
        ("Proceedings of the Conference.", True),
        ("Brown, Author. 2021. Real Paper Title. Venue.", False),
    ])
    def test_parametrized_fragment_detection(self, fragment_start: str, should_merge: bool):
        anchor = "Author, A. and Author, B. 2020. Paper Title. In"
        refs = [anchor, fragment_start]
        result = _merge_fragments(refs)
        if should_merge:
            assert len(result) == 1
        else:
            assert len(result) == 2
