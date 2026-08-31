"""Integration tests for `natu draw`, driven through the real CLI
end-to-end via natu.cli.main().

Both drawing modes are exercised by chaining off a real natu align /
natu cluster run within the same test, rather than a hand-built fixture
file -- a true end-to-end pipeline check, at the cost of also depending on
those commands' own correctness (covered separately in test_align.py /
test_cluster.py).
"""

from __future__ import annotations

import pytest


class TestDraw:
    def test_msa_mode_draws_an_svg_from_a_real_align_output(self, tmp_path, run_natu, write_fasta):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq1", ["alanine", "glycine", "serine"]),
            ("seq2", ["alanine", "glycine", "threonine"]),
        ])
        aligned_fasta = tmp_path / "aligned.fasta"
        run_natu(["align", "-m", "match_mismatch", "-f", str(fasta), "-o", str(aligned_fasta)])

        svg_out = tmp_path / "out.svg"
        run_natu(["draw", "-m", str(aligned_fasta), "-o", str(svg_out)])

        content = svg_out.read_text()
        assert content.startswith("<svg")
        assert content.rstrip().endswith("</svg>")

    def test_network_mode_draws_an_html_page_from_a_real_cluster_output(
        self, tmp_path, run_natu, write_fasta
    ):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq_a", ["alanine", "glycine", "serine"]),
            ("seq_b", ["alanine", "glycine", "serine"]),
            ("seq_c", ["proline", "leucine", "proline"]),
        ])
        cluster_out = tmp_path / "cluster_out"
        run_natu([
            "cluster", "-m", "match_mismatch",
            "-f", str(fasta), "-o", str(cluster_out), "-t", "0.9",
        ])

        html_out = tmp_path / "out.html"
        run_natu(["draw", "-n", str(cluster_out / "network.graphml"), "-o", str(html_out)])

        content = html_out.read_text()
        assert content.startswith("<!DOCTYPE html>")
        assert "NATU Similarity Network" in content

    def test_network_mode_with_highlight_embeds_the_highlighted_sequence(
        self, tmp_path, run_natu, write_fasta
    ):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [
            ("seq_a", ["alanine", "glycine", "serine"]),
            ("seq_b", ["alanine", "glycine", "serine"]),
            ("seq_c", ["proline", "leucine", "proline"]),
        ])
        cluster_out = tmp_path / "cluster_out"
        run_natu([
            "cluster", "-m", "match_mismatch",
            "-f", str(fasta), "-o", str(cluster_out), "-t", "0.9",
        ])

        html_out = tmp_path / "out.html"
        run_natu([
            "draw", "-n", str(cluster_out / "network.graphml"),
            "-o", str(html_out), "-H", "alanine|glycine|serine",
        ])

        assert "alanine|glycine|serine" in html_out.read_text()

    def test_msa_mode_rejects_network_only_flags(self, tmp_path, run_natu, write_fasta):
        fasta = tmp_path / "in.fasta"
        write_fasta(fasta, [("seq1", ["alanine", "glycine"])])
        aligned_fasta = tmp_path / "aligned.fasta"
        run_natu(["align", "-m", "match_mismatch", "-f", str(fasta), "-o", str(aligned_fasta)])

        with pytest.raises(ValueError, match="--highlight is only valid together with --network"):
            run_natu([
                "draw", "-m", str(aligned_fasta), "-o", str(tmp_path / "out.svg"),
                "-H", "alanine|glycine",
            ])
