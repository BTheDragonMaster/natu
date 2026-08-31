"""Shared fixtures and test-environment setup for the NATU test suite.

Layout note: this ``tests/`` package sits next to ``src/`` at the repo root
(the standard layout for a hatchling ``src``-layout project). Run the suite
from the repo root with ``pytest`` (an editable install, e.g.
``pip install -e ".[dev]"``, makes ``natu`` importable; no path hacking is
needed here as long as that's in place).
"""

from __future__ import annotations

import numpy as np
import pytest
from Bio.Align import PairwiseAligner, substitution_matrices

from natu.aligner import setup_aligner
from natu.pairwise import Converter

# Note: natu.variants used to have a self-referential dataclass field
# annotation (`base: Variant | None`, evaluated eagerly with no
# `from __future__ import annotations`) that raised NameError on import for
# anything downstream of it (natu.cli -> natu.scoring -> natu.matrix ->
# natu.variants). That's now fixed (the annotation is quoted as
# `"Variant | None"`), confirmed by natu.variants importing cleanly, so the
# stub workaround that used to live here has been removed.


# --- Shared alignment fixtures -------------------------------------------------
#
# A tiny 4-symbol alphabet ("p", "q", "r", "s" -- standing in for arbitrary
# monomer names) with a simple match/mismatch substitution matrix, used
# anywhere a test needs a real Bio.Align.PairwiseAligner without pulling in
# NATU's packaged chemistry-based matrices.

ALPHABET = ("p", "q", "r", "s")
MATCH_SCORE = 4.0
MISMATCH_SCORE = -1.0


@pytest.fixture
def converter() -> Converter:
    """Converter mapping ALPHABET symbols to sequential ints (and back)."""
    index = {symbol: i for i, symbol in enumerate(ALPHABET)}
    reverse = {i: symbol for symbol, i in index.items()}
    return Converter(
        to_identifier=lambda item: np.int32(index[item]),
        from_identifier=lambda ident: reverse[int(ident)],
    )


def _substitution_matrix() -> substitution_matrices.Array:
    n = len(ALPHABET)
    data = np.full((n, n), MISMATCH_SCORE, dtype=np.float64)
    np.fill_diagonal(data, MATCH_SCORE)
    return substitution_matrices.Array(ALPHABET, 2, data, np.float64)


@pytest.fixture
def substitution_matrix() -> substitution_matrices.Array:
    return _substitution_matrix()


@pytest.fixture
def make_aligner():
    """
    Factory fixture: ``make_aligner(mode="global", **gap_kwargs)`` builds a
    real ``PairwiseAligner`` over the shared test alphabet via
    ``natu.aligner.setup_aligner``, so tests exercise NATU's own aligner
    setup code rather than a hand-rolled one.
    """

    def _make(mode: str = "global", **gap_kwargs) -> PairwiseAligner:
        return setup_aligner(_substitution_matrix(), mode=mode, **gap_kwargs)

    return _make
