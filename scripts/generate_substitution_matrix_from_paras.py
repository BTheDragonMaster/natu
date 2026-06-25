"""
Build a BLOSUM-style log-odds substitution matrix from PARAS predictions
confidence vectors over substrates.

Pipeline:
    matrix_from_paras_results(...)  ->  (substrates, matrix)   [n_substrates x n_domains]
    background_frequencies(matrix) ->  q                       [n_substrates]
    joint_frequencies(matrix)      ->  p                        [n_substrates x n_substrates]
    log_odds_matrix(p, q)          ->  S                        [n_substrates x n_substrates]

The intermediate quantities (p, q, raw Gram matrix) are returned alongside
S so you can sanity-check them before trusting the final scores.
"""

from enum import Enum
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
import logging

import numpy as np
from numpy.typing import NDArray
import pandas as pd

logger = logging.getLogger(__name__)

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Build a BLOSUM-style log-odds substitution matrix from PARAS predictions")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to PARAS predictions file")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to output file")
    parser.add_argument("-c", "--confidence_threshold", default=0.4, type=float, help="Confidence threshold")

    args = parser.parse_args()
    return args


@dataclass
class SubstitutionMatrixResult:
    substrates: list[str]
    scores: NDArray            # S_ij, the BLOSUM-style log-odds matrix
    raw_joint: NDArray         # un-normalized co-confidence (Gram matrix)
    p: NDArray                 # normalized joint frequencies, sums to 1
    q: NDArray                 # normalized marginal/background frequencies, sums to 1
    pseudocount: float
    scale: float

    def write_substitution_matrix(self, out_file: str) -> None:
        df = pd.DataFrame(self.scores, index=self.substrates, columns=self.substrates)
        df.to_csv(out_file, sep="\t", index=True)

def background_frequencies(matrix: NDArray) -> NDArray:
    """
    Marginal confidence mass per substrate, normalized to sum to 1.

    Parameters
    ----------
    matrix : NDArray
        Shape (n_substrates, n_domains). Raw confidence scores
        (NOT cosine-normalized — magnitude matters here).

    Returns
    -------
    NDArray, shape (n_substrates,)
    """
    totals = matrix.sum(axis=1)  # total confidence mass per substrate, across all domains
    return totals / totals.sum()


def joint_frequencies(matrix: NDArray) -> tuple[NDArray, NDArray]:
    """
    Pairwise co-confidence between substrates, normalized to sum to 1.

    This is the raw Gram matrix (matrix @ matrix.T), which is the
    "how often do these two substrates both get confidence mass on the
    same domain" signal — analogous to counting aligned residue pairs
    in classic BLOSUM construction.

    Returns
    -------
    raw_joint : NDArray, shape (n_substrates, n_substrates)
        Unnormalized Gram matrix, for inspection.
    p : NDArray, shape (n_substrates, n_substrates)
        raw_joint normalized so the full matrix sums to 1.
    """
    raw_joint = matrix @ matrix.T
    p = raw_joint / raw_joint.sum()
    return raw_joint, p


def log_odds_matrix(
    p: NDArray,
    q: NDArray,
    pseudocount: float = 1e-9,
    scale: float = 2.0,
) -> NDArray:
    """
    Classic BLOSUM-style log-odds: S_ij = scale * log( p_ij / (q_i * q_j) ).

    Parameters
    ----------
    p : NDArray, shape (n, n)
        Normalized joint frequencies (sums to 1).
    q : NDArray, shape (n,)
        Normalized background/marginal frequencies (sums to 1).
    pseudocount : float
        Added to p before the log, to avoid log(0) for substrate pairs
        with ~zero observed joint confidence. Should be small relative
        to the smallest nonzero entries in p -- see diagnostics in
        build_substitution_matrix for a sane default check.
    scale : float
        Multiplicative constant (BLOSUM's 1/lambda, inverted). Purely
        affects dynamic range / how "spread out" the scores are; it does
        not change the sign or relative ordering of any entry. Tune by
        eye so scores land in a range you find readable (BLOSUM62 uses
        roughly [-4, +11]).

    Returns
    -------
    NDArray, shape (n, n)
    """
    background = np.outer(q, q)
    ratio = (p + pseudocount) / (background + pseudocount)
    return scale * np.log(ratio)


def reliability_scores(matrix: NDArray) -> NDArray:
    """
    Per-substrate reliability: how confidently and consistently this
    substrate is predicted, independent of how common it is.

    This is deliberately NOT frequency-corrected (unlike the log-odds
    diagonal you'd get from p_ii / q_i^2) -- it directly answers "when
    this substrate shows up, how confident is the model?" rather than
    "is this substrate's self-overlap surprising given how rare it is?"
    Rare substrates should NOT get an inflated score just for being rare.

    Defined as the mean confidence among domains where the substrate
    is the (or a) top-scoring call -- i.e. mean confidence conditional
    on the substrate actually being predicted, not averaged over all
    800k domains (which would just reflect how often it's predicted,
    not how confidently).

    Returns
    -------
    NDArray, shape (n_substrates,)
        Values in roughly [0, 1], same scale as your input confidences.
    """
    reliabilities = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        nonzero = matrix[i, matrix[i] > 0]
        reliabilities[i] = nonzero.mean() if nonzero.size else 0.0
    return reliabilities


def build_substitution_matrix(
    substrates: list[str],
    matrix: NDArray,
    pseudocount: float | None = None,
    scale: float = 2.0,
    diagonal_mode: str = "reliability",
    enforce_diagonal_dominance: bool = True,
) -> SubstitutionMatrixResult:
    """
    Full pipeline: raw confidence matrix -> BLOSUM-style substitution matrix.

    Parameters
    ----------
    substrates : list[str]
        Substrate names, in the row order of `matrix` (i.e. whatever
        matrix_from_paras_results gave you).
    matrix : NDArray
        Shape (n_substrates, n_domains). Raw confidence scores.
        Do NOT pass a cosine-normalized matrix here -- magnitude is
        exactly the signal that makes the diagonal non-flat.
    pseudocount : float, optional
        If None, auto-set to 1e-3 times the smallest nonzero entry of p.
        This keeps the pseudocount from dominating real signal while
        still preventing log(0).
    scale : float
        See log_odds_matrix.
    diagonal_mode : str
        "log_odds" -- pure BLOSUM formula, S_ii = scale * log(p_ii / q_i^2).
            WARNING: this rewards RARE substrates with high diagonal
            scores (self-overlap looks "surprising" relative to a small
            background frequency), even if their absolute confidence is
            low. Almost certainly not what you want here -- see the
            worked example in the accompanying test file.
        "reliability" (default) -- diagonal replaced with a directly
            confidence-driven score: high mean confidence where the
            substrate is actually called -> high diagonal, independent
            of how common the substrate is. Off-diagonal entries are
            still pure log-odds, so negative scores for
            unrelated/anti-correlated substrates are preserved.
    enforce_diagonal_dominance : bool
        If True (default) and diagonal_mode == "reliability", clamp
        each diagonal entry up to at least the largest off-diagonal
        value in its row, so no substrate pair can outscore a
        substrate matching itself. Real BLOSUM matrices always satisfy
        this; the raw reliability score doesn't guarantee it on its
        own (two substrates that are very frequently confused with
        each other can otherwise score higher than either one's own
        diagonal). Has no effect when diagonal_mode == "log_odds",
        which is already internally consistent.

    Returns
    -------
    SubstitutionMatrixResult
    """
    if matrix.ndim != 2 or matrix.shape[0] != len(substrates):
        raise ValueError(
            f"matrix shape {matrix.shape} doesn't match {len(substrates)} substrates "
            "-- did you pass matrix.T by mistake?"
        )
    if diagonal_mode not in ("log_odds", "reliability"):
        raise ValueError(f"unknown diagonal_mode: {diagonal_mode!r}")

    q = background_frequencies(matrix)
    raw_joint, p = joint_frequencies(matrix)

    if pseudocount is None:
        nonzero = p[p > 0]
        pseudocount = float(nonzero.min()) * 1e-3 if nonzero.size else 1e-9

    scores = log_odds_matrix(p, q, pseudocount=pseudocount, scale=scale)

    if diagonal_mode == "reliability":
        reliab = reliability_scores(matrix)
        # Rescale reliability (roughly [0,1]) onto the same dynamic range
        # as the off-diagonal log-odds scores, so the diagonal doesn't
        # look out of place next to e.g. BLOSUM62-style [-4, +11] values.
        off_diag_mask = ~np.eye(len(substrates), dtype=bool)
        off_diag_max = scores[off_diag_mask].max() if off_diag_mask.any() else 1.0
        # Map reliability in [0, 1] onto [0, off_diag_max * 1.5], so a
        # perfectly-confident substrate scores somewhat above the
        # strongest off-diagonal positive score, the way BLOSUM62's
        # diagonal sits above its off-diagonal entries.
        target_max = max(off_diag_max * 1.5, 1e-6)
        diag_scores = reliab * target_max
        np.fill_diagonal(scores, diag_scores)

        if enforce_diagonal_dominance:
            # Guarantee S_ii >= max(S_ij for j != i) for every row, the
            # way real BLOSUM matrices behave. Without this, a substrate
            # pair that's heavily confused with each other (high
            # off-diagonal log-odds) can outscore one or both of their
            # own, less-reliable diagonals -- see the ConfusedPairA/B
            # example in test_blosum_style.py. This only ever raises a
            # diagonal that's too low; it never lowers anything.
            n = len(substrates)
            for i in range(n):
                row_off_diag_max = np.delete(scores[i], i).max()
                if scores[i, i] <= row_off_diag_max:
                    scores[i, i] = row_off_diag_max + 1e-6

    return SubstitutionMatrixResult(
        substrates=substrates,
        scores=scores,
        raw_joint=raw_joint,
        p=p,
        q=q,
        pseudocount=pseudocount,
        scale=scale,
    )


def diagonal_report(result: SubstitutionMatrixResult, top_n: int = 10) -> str:
    """
    Quick human-readable check: are diagonal values actually varying,
    and does that variation track marginal frequency the way we expect?
    """
    diag = np.diag(result.scores)
    order = np.argsort(diag)[::-1]
    lines = ["substrate\tdiag_score\tbackground_freq_q"]
    for i in order[:top_n]:
        lines.append(f"{result.substrates[i]}\t{diag[i]:.3f}\t{result.q[i]:.5f}")
    lines.append("...")
    for i in order[-top_n:]:
        lines.append(f"{result.substrates[i]}\t{diag[i]:.3f}\t{result.q[i]:.5f}")
    return "\n".join(lines)


@dataclass
class ParasPrediction:
    substrate: str
    confidence: float


class SimilarityMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


class ParasResult:
    def __init__(self, domain_id: str, predictions: list[ParasPrediction]) -> None:
        self.domain_id = domain_id
        self.predictions = predictions

    @classmethod
    def from_file_line(cls, line: str):
        prediction_data = line.split("\t")
        domain_id = prediction_data[0]
        predictions_and_confidences = prediction_data[1:]
        predictions = []
        confidences = []
        for i, prediction_or_confidence in enumerate(predictions_and_confidences):
            if i % 2 == 0:
                predictions.append(prediction_or_confidence)
            else:
                confidences.append(float(prediction_or_confidence))

        assert len(predictions) == len(confidences)

        paras_predictions = []

        for i, prediction in enumerate(predictions):
            confidence = confidences[i]
            paras_predictions.append(ParasPrediction(prediction, confidence))

        return cls(domain_id, paras_predictions)


def parse_paras_results(paras_results_file: str, confidence_threshold: float = 0.4) -> list[ParasResult]:
    paras_results: list[ParasResult] = []
    with open(paras_results_file, "r") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if line:
                paras_result = ParasResult.from_file_line(line)
                if paras_result.predictions[0].confidence >= confidence_threshold:
                    paras_results.append(paras_result)

    return paras_results


def matrix_from_paras_results(paras_results: list[ParasResult]) -> tuple[list[str], NDArray]:
    if not paras_results:
        return [], np.array([])

    for paras_result in paras_results:
        paras_result.predictions.sort(key=lambda p: p.substrate)

    substrates = [p.substrate for p in paras_results[0].predictions]

    n_substrates = len(substrates)
    n_domains = len(paras_results)

    matrix = np.zeros((n_substrates, n_domains), dtype=np.float32)

    for domain_i, result in enumerate(paras_results):
        for substrate_i, prediction in enumerate(result.predictions):
            matrix[substrate_i, domain_i] = prediction.confidence

    return substrates, matrix


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    logger.info("Parsing PARAS results..")
    paras_results = parse_paras_results(args.input, args.confidence_threshold)
    logger.info(f"Domains kept: {len(paras_results)}")
    logger.info("Converting results into matrix..")
    substrates, matrix = matrix_from_paras_results(paras_results)
    logger.info("Building substitution matrix..")
    substitution_matrix = build_substitution_matrix(substrates, matrix)
    logger.info("Writing substitution matrix..")
    substitution_matrix.write_substitution_matrix(args.output)

if __name__ == "__main__":
    main()
