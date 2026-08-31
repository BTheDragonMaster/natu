"""Integration tests for `natu align`, driven through the real CLI
end-to-end via natu.cli.main().
"""

from __future__ import annotations

import pytest


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
