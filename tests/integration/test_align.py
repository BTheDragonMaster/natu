"""Integration tests for `natu align`, driven through the real CLI
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


def _parse_aligned_fasta(text: str) -> dict[str, tuple[float, list[str]]]:
    """header -> (alignment_score, aligned monomer tokens)."""
    records: dict[str, tuple[float, list[str]]] = {}
    lines = [line for line in text.splitlines() if line]
    for i in range(0, len(lines), 2):
        header_line = lines[i]
        header, _, score_part = header_line[1:].partition("|alignment_score=")
        records[header] = (float(score_part), lines[i + 1].split("|"))
    return records


class TestAlign:
    def test_aligns_same_length_sequences_with_match_mismatch(self, tmp_path, run_natu, write_fasta):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq1", ["alanine", "glycine", "serine"]),
            ("seq2", ["alanine", "glycine", "threonine"]),
            ("seq3", ["alanine", "leucine", "serine"]),
        ])
        out = tmp_path / "out.fasta"

        run_natu(["align", "-m", "match_mismatch", "-f", str(fasta), "-o", str(out)])

        records = _parse_aligned_fasta(out.read_text())
        assert set(records) == {"seq1", "seq2", "seq3"}

        # seq1 aligned against itself as the reference: every monomer matches.
        score1, tokens1 = records["seq1"]
        assert score1 == pytest.approx(3.0)
        assert tokens1 == ["alanine", "glycine", "serine"]

        # seq2 and seq3 each share 2 of 3 monomers with seq1, no reason for
        # a global alignment of equal-length sequences to need any gaps.
        for header in ("seq2", "seq3"):
            score, tokens = records[header]
            assert score == pytest.approx(2.0)
            assert len(tokens) == 3
            assert "GAP" not in tokens

    def test_raises_for_a_monomer_outside_the_matrix_alphabet(self, tmp_path, run_natu, write_fasta):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [("seq1", ["not_a_real_monomer"])])
        out = tmp_path / "out.fasta"

        with pytest.raises(ValueError, match="not_a_real_monomer"):
            run_natu(["align", "-m", "match_mismatch", "-f", str(fasta), "-o", str(out)])

    def test_smiles_flag_expands_the_alphabet_with_tailoring_variants(
        self, tmp_path, run_natu, write_fasta, real_smiles_file
    ):
        """A real D-form variant name -- not a literal entry in the
        packaged match_mismatch matrix, and not itself a row in
        smiles.tsv -- only becomes a usable monomer once --smiles triggers
        real tailoring-aware expansion (expand_substitution_matrix ->
        get_structure_variants) over the complete real substrate list.
        Confirmed by actually running both the without- and with-`-s`
        cases first, not assumed.
        """
        novel_variant = "(2R,3R)-2-amino-3-hydroxy-4-(4-nitrophenyl)butanoic acid"
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq1", ["alanine", "glycine"]),
            ("seq2", [novel_variant, "glycine"]),
        ])

        out_without = tmp_path / "without.fasta"
        with pytest.raises(ValueError, match="not found in the substitution matrix alphabet"):
            run_natu(["align", "-m", "match_mismatch", "-f", str(fasta), "-o", str(out_without)])

        out_with = tmp_path / "with.fasta"
        run_natu([
            "align", "-m", "match_mismatch",
            "-s", str(real_smiles_file),
            "-f", str(fasta), "-o", str(out_with),
        ])

        records = _parse_aligned_fasta(out_with.read_text())
        assert novel_variant in records["seq2"][1]

    def test_accepts_a_path_to_a_custom_substitution_matrix_file(
        self, tmp_path, run_natu, write_fasta
    ):
        """-m also accepts a path to a custom substitution matrix file, not
        just a built-in name. seq1/seq2 differ at exactly one (non-shared)
        position, so a straight substitution is the only sensible edit --
        no ambiguity from gap-driven realignment, confirmed by actually
        running this first."""
        matrix_file = tmp_path / "custom_matrix.tsv"
        _write_custom_matrix(matrix_file, [
            ("alanine", [100.0, 3.0, 3.0]),
            ("glycine", [3.0, 100.0, 3.0]),
            ("threonine", [3.0, 3.0, 100.0]),
        ])

        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq1", ["alanine", "glycine"]),
            ("seq2", ["alanine", "threonine"]),
        ])
        out = tmp_path / "out.fasta"

        run_natu(["align", "-m", str(matrix_file), "-f", str(fasta), "-o", str(out)])

        records = _parse_aligned_fasta(out.read_text())
        score1, tokens1 = records["seq1"]
        assert score1 == pytest.approx(200.0)  # self-alignment: 100 + 100
        assert tokens1 == ["alanine", "glycine"]

        score2, tokens2 = records["seq2"]
        assert score2 == pytest.approx(103.0)  # alanine-alanine (100) + threonine-glycine (3)
        assert tokens2 == ["alanine", "threonine"]

    def test_progressive_flag_uses_a_upgma_guide_tree_instead_of_a_center_star(
        self, tmp_path, run_natu, write_fasta
    ):
        """-p switches from center-star to progressive (UPGMA guide-tree)
        MSA. seq1/seq2 share 2 of 3 monomers and seq3 shares none, so the
        guide tree merges seq1/seq2 first and seq3 last -- confirmed by
        actually running this first: unlike center-star (which always puts
        the chosen center in row 0), progressive orders rows by tree-merge
        order, here starting with the odd one out, seq3."""
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq1", ["alanine", "glycine", "serine"]),
            ("seq2", ["alanine", "glycine", "threonine"]),
            ("seq3", ["proline", "leucine", "proline"]),
        ])
        out = tmp_path / "out.fasta"

        run_natu(["align", "-m", "match_mismatch", "-p", "-f", str(fasta), "-o", str(out)])

        records = _parse_aligned_fasta(out.read_text())
        assert list(records) == ["seq3", "seq2", "seq1"]

        score3, tokens3 = records["seq3"]
        assert score3 == pytest.approx(3.0)  # self-alignment: 3 matches
        assert tokens3 == ["proline", "leucine", "proline"]

        for header in ("seq1", "seq2"):
            score, tokens = records[header]
            assert score == pytest.approx(0.0)  # no monomers shared with seq3
            assert "GAP" not in tokens
