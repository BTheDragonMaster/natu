"""Integration tests for `natu build`, driven through the real CLI
end-to-end via natu.cli.main().

Covers argument wiring and file I/O; the underlying chemistry (Jaccard/
Tanimoto similarity rescaling, PARAS reference matching) is already
covered at the unit level in test_matrix.py's TestBuildSubstitutionMatrix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Real substrate SMILES, copied verbatim from
# src/natu/data/structures/smiles.tsv (same source used throughout the suite).
ALANINE = "C[C@@H](C(=O)O)N"
GLYCINE = "NCC(=O)O"
SERINE = "C([C@@H](C(=O)O)N)O"


def _write_smiles_file(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["substrate\tsmiles"] + [f"{name}\t{smi}" for name, smi in rows]
    path.write_text("\n".join(lines) + "\n")


class TestBuild:
    def test_builds_an_ecfp_matrix_from_real_smiles(self, tmp_path, run_natu):
        smiles_file = tmp_path / "smiles.tsv"
        _write_smiles_file(smiles_file, [("alanine", ALANINE), ("glycine", GLYCINE), ("serine", SERINE)])
        out_file = tmp_path / "matrix.txt"

        run_natu(["build", "-s", str(smiles_file), "-m", "ecfp", "-o", str(out_file)])

        result = pd.read_csv(out_file, sep="\t", index_col=0)
        assert list(result.index) == ["alanine", "glycine", "serine"]
        assert list(result.columns) == ["alanine", "glycine", "serine"]
        # rescale_similarity_matrix's target_max: every self-pair hits it,
        # confirmed directly in test_matrix.py.
        for name in result.index:
            assert result.loc[name, name] == pytest.approx(7.0)

    def test_builds_a_paras_based_matrix_using_the_real_reference_dataset(self, tmp_path, run_natu):
        """The deepest chemistry path build_substitution_matrix has:
        matching real query substrates against the packaged PARAS
        reference set (src/natu/data/structures/paras_smiles.tsv +
        src/natu/data/substitution_matrices/paras_based.txt)."""
        smiles_file = tmp_path / "smiles.tsv"
        _write_smiles_file(smiles_file, [("alanine", ALANINE), ("glycine", GLYCINE)])
        out_file = tmp_path / "matrix.txt"

        run_natu(["build", "-s", str(smiles_file), "-m", "paras_based", "-o", str(out_file)])

        result = pd.read_csv(out_file, sep="\t", index_col=0)
        assert list(result.index) == ["alanine", "glycine"]
        assert result.isna().sum().sum() == 0

    def test_defaults_to_ecfp_when_matrix_type_is_omitted(self, tmp_path, run_natu):
        smiles_file = tmp_path / "smiles.tsv"
        _write_smiles_file(smiles_file, [("alanine", ALANINE), ("glycine", GLYCINE)])
        out_explicit = tmp_path / "explicit.txt"
        out_default = tmp_path / "default.txt"

        run_natu(["build", "-s", str(smiles_file), "-m", "ecfp", "-o", str(out_explicit)])
        run_natu(["build", "-s", str(smiles_file), "-o", str(out_default)])

        pd.testing.assert_frame_equal(
            pd.read_csv(out_explicit, sep="\t", index_col=0),
            pd.read_csv(out_default, sep="\t", index_col=0),
        )
