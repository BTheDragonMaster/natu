"""Tests for the pure filtering/parsing helpers in natu.cli:
process_sequences, filter_by_min_length, filter_by_max_unknown_fraction,
deep_update, load_config, and read_monomer_fasta.

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from natu.cli import (
    deep_update,
    filter_by_max_unknown_fraction,
    filter_by_min_length,
    load_config,
    process_sequences,
    read_monomer_fasta,
)
from natu.constants import SubstitutionMatrix


class TestProcessSequences:
    def test_is_a_noop_when_wildcard_for_unknowns_is_off(self):
        """
        Documented gotcha: wildcard.wildcard_character can be *named* in
        config even while wildcard_for_unknowns is False -- in that case
        process_sequences must leave unrecognized monomers completely
        alone.
        """
        config = {"wildcard": {"wildcard_for_unknowns": False, "wildcard_character": "X"}}
        result = process_sequences(["ala", "gly"], [["ala", "not_in_alphabet"]], config)
        assert result == [["ala", "not_in_alphabet"]]

    def test_replaces_unknowns_when_enabled(self):
        config = {"wildcard": {"wildcard_for_unknowns": True, "wildcard_character": "X"}}
        result = process_sequences(["ala", "gly"], [["ala", "not_in_alphabet"]], config)
        assert result == [["ala", "X"]]

    def test_on_empty_sequence_list(self):
        config = {"wildcard": {"wildcard_for_unknowns": True, "wildcard_character": "X"}}
        assert process_sequences(["ala"], [], config) == []


class TestFilterByMinLength:
    def test_drops_sequences_shorter_than_minimum(self):
        headers = ["a", "b", "c"]
        sequences = [["p"], ["p", "q"], ["p", "q", "r"]]
        kept_headers, kept_sequences = filter_by_min_length(headers, sequences, min_length=2)
        assert kept_headers == ["b", "c"]
        assert kept_sequences == [["p", "q"], ["p", "q", "r"]]

    def test_zero_keeps_everything_including_empty_sequences(self):
        kept_headers, kept_sequences = filter_by_min_length(["a"], [[]], min_length=0)
        assert kept_headers == ["a"]
        assert kept_sequences == [[]]

    def test_preserves_relative_order(self):
        headers = ["a", "b", "c", "d"]
        sequences = [["p", "p"], ["p"], ["p", "p", "p"], ["p"]]
        kept_headers, _ = filter_by_min_length(headers, sequences, min_length=2)
        assert kept_headers == ["a", "c"]


class TestFilterByMaxUnknownFraction:
    def test_drops_sequences_over_threshold(self):
        headers = ["a", "b"]
        sequences = [["X", "X", "p"], ["p", "p", "p"]]
        kept_headers, kept_sequences = filter_by_max_unknown_fraction(
            headers, sequences, wildcard_character="X", max_fraction=0.5
        )
        assert kept_headers == ["b"]
        assert kept_sequences == [["p", "p", "p"]]

    def test_max_fraction_at_or_above_one_is_a_shortcut(self):
        """max_fraction >= 1.0 must keep everything without even needing a
        wildcard_character -- this is the escape hatch for configs that
        don't define one at all."""
        headers, sequences = filter_by_max_unknown_fraction(
            ["a"], [["p"]], wildcard_character=None, max_fraction=1.0
        )
        assert headers == ["a"]
        assert sequences == [["p"]]

    def test_requires_wildcard_character_below_one(self):
        with pytest.raises(ValueError, match="does not define a wildcard"):
            filter_by_max_unknown_fraction(["a"], [["p"]], wildcard_character=None, max_fraction=0.1)

    def test_keeps_empty_sequences(self):
        """An empty sequence has no monomers to divide by -- must be
        handled as a special case rather than raising ZeroDivisionError."""
        headers, sequences = filter_by_max_unknown_fraction(
            ["a"], [[]], wildcard_character="X", max_fraction=0.0
        )
        assert headers == ["a"]
        assert sequences == [[]]

    def test_is_a_silent_noop_when_config_has_wildcard_disabled(self):
        """
        Documented gotcha from the project reference: this filter counts
        the literal wildcard character, which only ever appears in a
        sequence if process_sequences() actually put it there (i.e.
        wildcard_for_unknowns was True). A sequence full of
        genuinely-unrecognized monomers that were NEVER replaced (because
        wildcard_for_unknowns was False) has zero occurrences of the
        wildcard character and is not filtered at all, no matter how
        "unknown" it really is.
        """
        headers, sequences = filter_by_max_unknown_fraction(
            ["a"], [["not_a_real_monomer", "also_not_real"]], wildcard_character="X", max_fraction=0.0
        )
        # Nothing equals "X" literally, so nothing is counted as unknown,
        # so nothing is dropped -- even at max_fraction=0.0.
        assert headers == ["a"]
        assert sequences == [["not_a_real_monomer", "also_not_real"]]


class TestDeepUpdate:
    def test_merges_nested_dicts_without_dropping_untouched_keys(self):
        base = {"a": {"b": 1, "c": 2}, "d": 5}
        update = {"a": {"b": 99}, "e": 7}
        result = deep_update(base, update)
        assert result == {"a": {"b": 99, "c": 2}, "d": 5, "e": 7}

    def test_replaces_non_dict_values_outright(self):
        base = {"a": {"b": 1}}
        update = {"a": "now a string"}
        assert deep_update(base, update) == {"a": "now a string"}

    def test_on_empty_update_returns_base_unchanged(self):
        base = {"a": 1}
        assert deep_update(base, {}) == {"a": 1}


class TestLoadConfig:
    def test_default_has_wildcard_section(self):
        config = load_config()
        assert "wildcard" in config

    def test_selects_matrix_specific_defaults(self):
        config = load_config(matrix_type=SubstitutionMatrix.ECFP)
        assert config["wildcard"]["wildcard_character"] == "X"

    def test_merges_user_overrides_onto_defaults(self, tmp_path: Path):
        user_config = tmp_path / "user.yaml"
        user_config.write_text("wildcard:\n  wildcard_for_unknowns: true\n")

        merged = load_config(config_file=user_config)

        assert merged["wildcard"]["wildcard_for_unknowns"] is True
        # untouched default keys must survive the merge
        assert "wildcard_character" in merged["wildcard"]

    def test_empty_user_file_falls_back_to_default(self, tmp_path: Path):
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        assert load_config(config_file=empty_config) == load_config()


class TestReadMonomerFasta:
    def test_happy_path(self, tmp_path: Path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">seq1 some description\nala|gly|ser\n>seq2\nleu|leu\n")

        records = read_monomer_fasta(fasta)

        assert records == [
            ("seq1 some description", ["ala", "gly", "ser"]),
            ("seq2", ["leu", "leu"]),
        ]

    def test_rejects_empty_sequence(self, tmp_path: Path):
        fasta = tmp_path / "empty_seq.fa"
        fasta.write_text(">seq1\n\n")

        with pytest.raises(ValueError, match="empty sequence"):
            read_monomer_fasta(fasta)

    def test_rejects_empty_monomer(self, tmp_path: Path):
        fasta = tmp_path / "empty_monomer.fa"
        fasta.write_text(">seq1\nala||ser\n")

        with pytest.raises(ValueError, match="empty monomer"):
            read_monomer_fasta(fasta)

    def test_rejects_sequence_before_any_header(self, tmp_path: Path):
        fasta = tmp_path / "no_header.fa"
        fasta.write_text("ala|gly\n")

        with pytest.raises(ValueError, match="sequence before first FASTA header"):
            read_monomer_fasta(fasta)

    def test_rejects_a_file_with_no_records(self, tmp_path: Path):
        fasta = tmp_path / "nothing.fa"
        fasta.write_text("")

        with pytest.raises(ValueError, match="no FASTA records found"):
            read_monomer_fasta(fasta)

    def test_correctly_parses_a_wrapped_multiline_sequence(self, tmp_path: Path):
        """
        Regression test for a fixed bug: read_monomer_fasta used to join a
        record's sequence lines with NO separator before splitting on "|",
        so a sequence wrapped across multiple FASTA lines got the monomer
        at the line break silently glued to the next line's first monomer
        (e.g. "ala|gly" + "ser|leu" -> ["ala", "glyser", "leu"]). It now
        joins lines with "|" first, so each line's monomers stay distinct.
        """
        fasta = tmp_path / "wrapped.fa"
        fasta.write_text(">seq1\nala|gly\nser|leu\n")

        records = read_monomer_fasta(fasta)

        assert records == [("seq1", ["ala", "gly", "ser", "leu"])]
