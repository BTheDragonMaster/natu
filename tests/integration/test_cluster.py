"""Integration tests for `natu cluster`, driven through the real CLI
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


def _parse_clusters(text: str) -> dict[str, str]:
    lines = text.splitlines()
    return dict(line.split("\t") for line in lines[1:] if line)


class TestCluster:
    def test_writes_network_graphml_edges_and_clusters_with_expected_grouping(
        self, tmp_path, run_natu, write_fasta
    ):
        """seq_a/seq_b and seq_d/seq_e are each identical pairs (similarity
        1.0); seq_c differs from seq_a/seq_b at one of three monomers
        (similarity 2/3); seq_a/b share nothing with seq_d/e. At cutoff
        0.9, this should produce exactly 3 clusters: {seq_a, seq_b},
        {seq_d, seq_e}, and {seq_c} alone -- confirmed by actually running
        this before writing the assertions.
        """
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq_a", ["alanine", "glycine", "serine"]),
            ("seq_b", ["alanine", "glycine", "serine"]),
            ("seq_c", ["alanine", "glycine", "threonine"]),
            ("seq_d", ["proline", "leucine", "proline"]),
            ("seq_e", ["proline", "leucine", "proline"]),
        ])
        out_dir = tmp_path / "cluster_out"

        run_natu([
            "cluster", "-m", "match_mismatch",
            "-f", str(fasta), "-o", str(out_dir), "-t", "0.9",
        ])

        assert sorted(p.name for p in out_dir.iterdir()) == ["clusters.tsv", "edges.tsv", "network.graphml"]

        clusters = _parse_clusters((out_dir / "clusters.tsv").read_text())
        assert clusters["seq_a"] == clusters["seq_b"]
        assert clusters["seq_d"] == clusters["seq_e"]
        assert clusters["seq_c"] not in (clusters["seq_a"], clusters["seq_d"])
        assert len(set(clusters.values())) == 3

        edges = (out_dir / "edges.tsv").read_text()
        assert edges.count("\n") == 3  # header + 2 edges
        assert "seq_a\tseq_b\t1.000000" in edges
        assert "seq_d\tseq_e\t1.000000" in edges

        graphml = (out_dir / "network.graphml").read_text()
        assert graphml.startswith("<?xml")
        assert "graphml" in graphml

    def test_min_length_drops_short_sequences_before_clustering(self, tmp_path, run_natu, write_fasta):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("long_one", ["alanine", "glycine", "serine"]),
            ("long_two", ["alanine", "glycine", "serine"]),
            ("too_short", ["alanine"]),
        ])
        out_dir = tmp_path / "cluster_out"

        run_natu([
            "cluster", "-m", "match_mismatch",
            "-f", str(fasta), "-o", str(out_dir), "-t", "0.9", "-l", "2",
        ])

        clusters = _parse_clusters((out_dir / "clusters.tsv").read_text())
        assert "too_short" not in clusters
        assert set(clusters) == {"long_one", "long_two"}

    def test_smiles_flag_lets_a_tailoring_variant_be_clustered(
        self, tmp_path, run_natu, write_fasta, real_smiles_file
    ):
        novel_variant = "(2R,3R)-2-amino-3-hydroxy-4-(4-nitrophenyl)butanoic acid"
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq1", ["alanine", "glycine"]),
            ("seq2", [novel_variant, "glycine"]),
        ])
        out_without = tmp_path / "cluster_out_without"

        with pytest.raises(ValueError, match="not found in the substitution matrix alphabet"):
            run_natu([
                "cluster", "-m", "match_mismatch",
                "-f", str(fasta), "-o", str(out_without), "-t", "0.0",
            ])

        out_with = tmp_path / "cluster_out_with"
        run_natu([
            "cluster", "-m", "match_mismatch",
            "-s", str(real_smiles_file),
            "-f", str(fasta), "-o", str(out_with), "-t", "0.0",
        ])

        clusters = _parse_clusters((out_with / "clusters.tsv").read_text())
        assert set(clusters) == {"seq1", "seq2"}

    def test_accepts_a_path_to_a_custom_substitution_matrix_file(
        self, tmp_path, run_natu, write_fasta
    ):
        """Same custom matrix as the align/search cases: c1/c2 are
        identical (similarity 1.0, same cluster), c3 differs at one
        position and lands in its own cluster -- confirmed against a real
        run before writing these assertions."""
        matrix_file = tmp_path / "custom_matrix.tsv"
        _write_custom_matrix(matrix_file, [
            ("alanine", [100.0, 3.0, 3.0]),
            ("glycine", [3.0, 100.0, 3.0]),
            ("threonine", [3.0, 3.0, 100.0]),
        ])

        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("c1", ["alanine", "glycine"]),
            ("c2", ["alanine", "glycine"]),
            ("c3", ["alanine", "threonine"]),
        ])
        out_dir = tmp_path / "cluster_out"

        run_natu([
            "cluster", "-m", str(matrix_file),
            "-f", str(fasta), "-o", str(out_dir), "-t", "0.9",
        ])

        clusters = _parse_clusters((out_dir / "clusters.tsv").read_text())
        assert clusters["c1"] == clusters["c2"]
        assert clusters["c3"] not in (clusters["c1"],)
        assert len(set(clusters.values())) == 2
