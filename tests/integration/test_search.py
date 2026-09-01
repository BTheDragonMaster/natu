"""Integration tests for `natu search`, driven through the real CLI
end-to-end via natu.cli.main().
"""

from __future__ import annotations

import pytest


def _write_custom_matrix(path, rows):
    """Write a tab-separated custom substitution matrix file: rows is a
    list of (name, [scores...]) in the same order as the header."""
    names = [name for name, _ in rows]
    lines = ["\t" + "\t".join(names)]
    for name, scores in rows:
        lines.append(name + "\t" + "\t".join(f"{s:.1f}" for s in scores))
    path.write_text("\n".join(lines) + "\n")


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

    def test_accepts_a_path_to_a_custom_substitution_matrix_file(
        self, tmp_path, run_natu, write_fasta
    ):
        """Same custom-matrix scenario as natu align/cluster: alanine/
        glycine/threonine with a big self-score and a small, uniform
        cross-score, confirmed against a real run before writing these
        assertions."""
        matrix_file = tmp_path / "custom_matrix.tsv"
        _write_custom_matrix(matrix_file, [
            ("alanine", [100.0, 3.0, 3.0]),
            ("glycine", [3.0, 100.0, 3.0]),
            ("threonine", [3.0, 3.0, 100.0]),
        ])

        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, [("q1", ["alanine", "glycine"])])

        subject_fasta = tmp_path / "subjects.fasta"
        write_fasta(subject_fasta, [
            ("s1", ["alanine", "threonine"]),  # 1 of 2 monomers match
            ("s2", ["alanine", "glycine"]),    # identical: highest score
        ])
        out = tmp_path / "out.tsv"

        run_natu([
            "search", "-m", str(matrix_file),
            "-q", str(query_fasta), "-f", str(subject_fasta),
            "-o", str(out), "-t", "-1000",
        ])

        rows = _parse_search_output(out.read_text())
        assert [row["subject"] for row in rows] == ["s2", "s1"]
        assert [row["bitscore"] for row in rows] == ["200.000", "103.000"]

    def test_alignment_mode_flag_is_parsed_as_an_enum_and_changes_the_score(
        self, tmp_path, run_natu, write_fasta
    ):
        """-a/--alignment_mode now goes through parse_alignment_mode (str ->
        AlignmentMode) instead of a plain string choice -- checked here by
        actually running all three modes against a query much shorter than
        the subject and confirming they score differently, exactly as
        documented: global penalizes the subject's unmatched tail, local
        ignores it entirely, and glocal (forcing the shorter query to align
        in full while leaving the longer subject's overhang free) scores
        the same as local here but reports only the aligned core, not the
        subject's full length. Exact scores confirmed by actually running
        this first."""
        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, [("q1", ["alanine", "glycine"])])

        subject_fasta = tmp_path / "subjects.fasta"
        write_fasta(subject_fasta, [
            ("s1", ["alanine", "glycine", "serine", "threonine", "leucine"]),
        ])

        results = {}
        for mode in ("global", "local", "glocal"):
            out = tmp_path / f"out_{mode}.tsv"
            run_natu([
                "search", "-m", "match_mismatch", "-a", mode,
                "-q", str(query_fasta), "-f", str(subject_fasta),
                "-o", str(out), "-t", "-100",
            ])
            results[mode] = _parse_search_output(out.read_text())[0]

        assert results["global"]["bitscore"] == "1.300"
        assert results["local"]["bitscore"] == "2.000"
        assert results["glocal"]["bitscore"] == "2.000"
        # glocal reports only the aligned core (query's own length), unlike
        # local/global which pad out to the subject's full length with GAP.
        assert results["glocal"]["s_aligned_s"] == "alanine|glycine"
        assert results["local"]["s_aligned_s"] == "alanine|glycine|serine|threonine|leucine"

    def test_rejects_an_alignment_mode_that_is_neither_a_name_nor_valid(self, tmp_path, run_natu, write_fasta):
        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, [("q1", ["alanine", "glycine"])])
        subject_fasta = tmp_path / "subjects.fasta"
        write_fasta(subject_fasta, [("s1", ["alanine", "glycine"])])

        with pytest.raises(SystemExit):
            run_natu([
                "search", "-m", "match_mismatch", "-a", "bananamode",
                "-q", str(query_fasta), "-f", str(subject_fasta),
                "-o", str(tmp_path / "out.tsv"),
            ])
