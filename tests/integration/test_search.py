"""Integration tests for `natu search`, driven through the real CLI
end-to-end via natu.cli.main().
"""

from __future__ import annotations

import pytest


def _parse_search_output(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


class TestSearch:
    def test_finds_every_subject_scoring_at_or_above_threshold_sorted_by_score(
        self, tmp_path, run_natu, write_fasta
    ):
        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, [("q1", ["alanine", "glycine", "serine"])])

        subject_fasta = tmp_path / "subjects.fasta"
        write_fasta(subject_fasta, [
            ("s1", ["alanine", "glycine", "serine"]),    # identical: highest score
            ("s2", ["alanine", "glycine", "threonine"]),  # 2/3 monomers match
            ("s3", ["proline", "leucine", "proline"]),    # no shared monomers
        ])
        out = tmp_path / "out.tsv"

        run_natu([
            "search", "-m", "match_mismatch",
            "-q", str(query_fasta), "-f", str(subject_fasta),
            "-o", str(out), "-t", "0.0",
        ])

        rows = _parse_search_output(out.read_text())
        # match_mismatch's mismatch score is 0.0, not negative, so all three
        # subjects clear a threshold of 0.0 -- confirmed by actually running
        # this rather than assumed.
        assert [row["subject"] for row in rows] == ["s1", "s2", "s3"]
        assert [row["bitscore"] for row in rows] == ["3.000", "2.000", "0.000"]

    def test_threshold_excludes_lower_scoring_subjects(self, tmp_path, run_natu, write_fasta):
        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, [("q1", ["alanine", "glycine", "serine"])])

        subject_fasta = tmp_path / "subjects.fasta"
        write_fasta(subject_fasta, [
            ("s1", ["alanine", "glycine", "serine"]),
            ("s2", ["alanine", "glycine", "threonine"]),
            ("s3", ["proline", "leucine", "proline"]),
        ])
        out = tmp_path / "out.tsv"

        run_natu([
            "search", "-m", "match_mismatch",
            "-q", str(query_fasta), "-f", str(subject_fasta),
            "-o", str(out), "-t", "1.5",
        ])

        rows = _parse_search_output(out.read_text())
        assert [row["subject"] for row in rows] == ["s1", "s2"]  # s3 (score 0.0) excluded

    def test_smiles_flag_lets_a_tailoring_variant_be_used_as_a_subject(
        self, tmp_path, run_natu, write_fasta, real_smiles_file
    ):
        """Same real tailoring-variant expansion as natu align/cluster,
        exercised here for a subject sequence rather than a query."""
        novel_variant = "(2R,3R)-2-amino-3-hydroxy-4-(4-nitrophenyl)butanoic acid"
        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, [("q1", ["alanine", "glycine"])])

        subject_fasta = tmp_path / "subjects.fasta"
        write_fasta(subject_fasta, [("s1", [novel_variant, "glycine"])])

        out_without = tmp_path / "without.tsv"
        with pytest.raises(ValueError, match="not found in the substitution matrix alphabet"):
            run_natu([
                "search", "-m", "match_mismatch",
                "-q", str(query_fasta), "-f", str(subject_fasta),
                "-o", str(out_without), "-t", "-100",
            ])

        out_with = tmp_path / "with.tsv"
        run_natu([
            "search", "-m", "match_mismatch",
            "-s", str(real_smiles_file),
            "-q", str(query_fasta), "-f", str(subject_fasta),
            "-o", str(out_with), "-t", "-100",
        ])

        rows = _parse_search_output(out_with.read_text())
        assert len(rows) == 1
        assert rows[0]["subject"] == "s1"
