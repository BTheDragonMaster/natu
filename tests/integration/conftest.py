"""Shared fixtures for NATU's CLI integration tests.

These tests call natu.cli.main() directly (with sys.argv patched) rather
than spawning a subprocess for each one: that drives the exact same
argparse parsing, command dispatch, and file I/O a real `natu ...` shell
invocation goes through, just without subprocess overhead or a dependency
on the console-script entry point being reinstalled after a source edit.

Every test collected under this directory is automatically marked
`@pytest.mark.integration` (see pytest_collection_modifyitems below), so
the whole suite can be selected with `pytest -m integration` or excluded
from a fast local run with `pytest -m "not integration"`.
"""

from __future__ import annotations

import importlib.resources as res
import sys
from pathlib import Path

import pytest

from natu.cli import main
from natu.network import sequence_to_label


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def run_natu(monkeypatch):
    """Factory fixture: run_natu(["align", "-m", "match_mismatch", ...]).

    Patches sys.argv (cli() reads it via argparse's default parse_args())
    and calls natu.cli.main() -- the real CLI entry point installed as the
    `natu` console script -- in-process.
    """

    def _run(args: list[str]) -> None:
        monkeypatch.setattr(sys, "argv", ["natu"] + args)
        main()

    return _run


@pytest.fixture
def write_fasta():
    """Factory fixture: write_fasta(path, [("header", ["monomer", "names"]), ...]).

    Writes a monomer FASTA using NATU's own canonical pipe-joined sequence
    label convention (natu.network.sequence_to_label), matching exactly
    what natu.cli.read_monomer_fasta expects on the other end.
    """

    def _write(path: Path, records: list[tuple[str, list[str]]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for header, monomers in records:
                handle.write(f">{header}\n{sequence_to_label(monomers)}\n")

    return _write


@pytest.fixture(scope="session")
def real_smiles_file():
    """Path to the real packaged substrate SMILES file
    (src/natu/data/structures/smiles.tsv) -- all 278 real NATU substrates.
    Used for the --smiles / tailoring-aware-expansion integration cases.

    Requires pikachu-chem>=1.1.6 (pinned in pyproject.toml): an
    alpha-carbon chirality-matching bug in 1.1.5 raised a PIKAChU
    StructureError inside get_d_structure for 5 of these 278 real
    substrates (2-methylserine, 2S-methyl-3-oxobutyrine, D-isovaline,
    isovaline, norcoronamic acid), which made --smiles unusable against
    the complete real substrate list at all. Fixed upstream in 1.1.6 and
    confirmed empirically (0 of 278 raise) before writing these tests.
    """
    with res.as_file(res.files("natu.data.structures") / "smiles.tsv") as path:
        yield Path(str(path))
