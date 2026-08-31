"""Pairwise sequence alignment module."""

from dataclasses import dataclass
from typing import TypeVar, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from natu.aligner import PairwiseAligner


T = TypeVar("T")


@dataclass(frozen=True)
class Converter:
    """
    Converter for sequences to be aligned.

    :cvar to_identifier: Function to convert items in sequences to integers for alignment.
    :cvar from_identifier: Function to convert integers back to items in sequences after alignment.
    :cvar gap_repr: Integer representation of a gap in the alignment (default: -1).
    :cvar insert_repr: Integer representation of an insertion in the alignment (default: -2).
    """

    to_identifier: Callable[[T], np.int32]
    from_identifier: Callable[[np.int32], T]

    gap_repr: np.int32 = np.int32(-1)
    insert_repr: np.int32 = np.int32(-2)

    def to_int_array(self, sequence: list[T]) -> NDArray[np.int32]:
        """
        Convert a sequence to a numpy array of integers for alignment.

        :param sequence: List of items in the sequence.
        :return: Numpy array of integers representing the sequence.
        """
        return np.array([self.to_identifier(item) for item in sequence], dtype=np.int32)

    def from_int_array(self, int_array: NDArray[np.int32]) -> list[T | None]:
        """
        Convert a numpy array of integers back to a sequence of items after alignment.

        :param int_array: Numpy array of integers representing the aligned sequence (with gap_repr for gaps).
        :return: List of items in the aligned sequence (with None for gaps).
        """
        return [self.from_identifier(item) if item != self.gap_repr else None for item in int_array]


def configure_glocal_end_gaps(
    aligner: PairwiseAligner,
    len_t: int,
    len_q: int,
    open_end_gap_score: float,
    extend_end_gap_score: float,
) -> None:
    """
    Configure a global-mode aligner's end-gap scores for ONE pair's "glocal" alignment.

    Glocal (as in BiG-SCAPE) forces the shorter of two sequences to be fully included in
    the alignment -- both its "arms" -- while letting the longer sequence's non-matching
    overhang go unpenalized, so a short sequence embedded inside a longer, otherwise
    unrelated one can still score well without being penalized for the longer sequence's
    extra length, while the short one can't get away with only partially matching either.

    Must be called again before every single pairwise alignment/score call made with this
    aligner, since which sequence is "longer" -- and therefore which side gets freed --
    changes from pair to pair; nothing here is sticky across calls.

    Biopython naming note, confirmed empirically (not from memory -- gap-score attribute
    names are easy to get backwards): a gap in seqB with unmatched residues left over in
    seqA is scored via the *deletion* attributes; a gap in seqA with unmatched residues
    left over in seqB is scored via the *insertion* attributes. NATU's own
    ``_pairwise_alignment``/``aligner.score`` calls always pass ``t`` as seqA and ``q`` as
    seqB, so:

    - ``t`` longer than ``q``: ``t``'s extra residues at the ends show up as gaps in
      ``q`` -> free the *deletion* end-gap scores; leave *insertion* end-gap scores at the
      configured (penalized) value, so ``q`` can't skip its own ends for free.
    - ``q`` longer than ``t``: the roles swap -> free *insertion*, leave *deletion* at the
      configured value.
    - Equal length: no side is "the longer one" to grant an overhang to -- both stay at
      the configured value, equivalent to plain global alignment for this pair.

    :param aligner: PairwiseAligner to mutate in place. Must already be in "global" mode
        (e.g. via ``natu.aligner.setup_aligner(..., mode="glocal")``).
    :param len_t: Length of the sequence that will be passed as ``t``/seqA.
    :param len_q: Length of the sequence that will be passed as ``q``/seqB.
    :param open_end_gap_score: The alignment config's normal (penalized) end-gap open
        score, applied to whichever side is not freed for this pair.
    :param extend_end_gap_score: As above, for the extend score.
    """
    if len_t > len_q:
        free, kept = 0.0, (open_end_gap_score, extend_end_gap_score)
        aligner.open_left_deletion_score = free
        aligner.extend_left_deletion_score = free
        aligner.open_right_deletion_score = free
        aligner.extend_right_deletion_score = free
        aligner.open_left_insertion_score = kept[0]
        aligner.extend_left_insertion_score = kept[1]
        aligner.open_right_insertion_score = kept[0]
        aligner.extend_right_insertion_score = kept[1]
    elif len_q > len_t:
        free, kept = 0.0, (open_end_gap_score, extend_end_gap_score)
        aligner.open_left_insertion_score = free
        aligner.extend_left_insertion_score = free
        aligner.open_right_insertion_score = free
        aligner.extend_right_insertion_score = free
        aligner.open_left_deletion_score = kept[0]
        aligner.extend_left_deletion_score = kept[1]
        aligner.open_right_deletion_score = kept[0]
        aligner.extend_right_deletion_score = kept[1]
    else:
        aligner.open_left_insertion_score = open_end_gap_score
        aligner.extend_left_insertion_score = extend_end_gap_score
        aligner.open_right_insertion_score = open_end_gap_score
        aligner.extend_right_insertion_score = extend_end_gap_score
        aligner.open_left_deletion_score = open_end_gap_score
        aligner.extend_left_deletion_score = extend_end_gap_score
        aligner.open_right_deletion_score = open_end_gap_score
        aligner.extend_right_deletion_score = extend_end_gap_score


def strip_glocal_free_overhang(
    t_a: NDArray[np.int32],
    q_a: NDArray[np.int32],
    gap_repr: np.int32,
    len_t: int,
    len_q: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Strip the longer sequence's free (zero-cost) overhang from a glocal alignment's output,
    so its length reflects only the genuinely-aligned core -- e.g. for
    ``natu.network.cluster_sequences``'s ``min_alignment_length`` check.

    Verified empirically that Biopython does not consistently represent a glocal
    alignment's free overhang the same way: sometimes it's excluded from
    ``alignment.coordinates`` entirely (like a local alignment's dangling ends, which
    ``_pairwise_alignment``'s existing ``trim`` handling already accounts for), and
    sometimes it's included as explicit leading/trailing gap columns within the
    coordinates (which ``trim`` does NOT strip, since it can't tell a real internal gap
    from a free one just by position). This function handles both cases the same way, by
    looking at the *shorter* sequence's own array: wherever the longer sequence overhangs
    for free, the shorter sequence has no residues there at all, so its array carries
    ``gap_repr`` at exactly those positions -- strip any such run from the start and/or
    end of the shorter side, and trim both arrays to match. A genuine internal gap in the
    shorter sequence (surrounded by real residues, not at the very edge) is left alone.

    :param t_a: Aligned ``t`` array, as returned by ``_pairwise_alignment``.
    :param q_a: Aligned ``q`` array, as returned by ``_pairwise_alignment``.
    :param gap_repr: Integer representation of a gap.
    :param len_t: Original (unaligned) length of the sequence passed as ``t``.
    :param len_q: Original (unaligned) length of the sequence passed as ``q``.
    :return: ``(t_a, q_a)``, both sliced to the same core region, with the shorter
        sequence's free leading/trailing overhang removed. Returned unchanged if
        ``len_t == len_q`` (glocal doesn't free either side in that case).
    """
    if len_t == len_q:
        return t_a, q_a

    shorter = t_a if len_t < len_q else q_a

    start = 0
    while start < len(shorter) and shorter[start] == gap_repr:
        start += 1

    end = len(shorter)
    while end > start and shorter[end - 1] == gap_repr:
        end -= 1

    return t_a[start:end], q_a[start:end]


def replace_unknowns_with_wildcards(alphabet: Iterable[str],
                                    sequence: list[str],
                                    wildcard_character: str) -> list[str]:

    """Replace unknowns with wildcard character in sequence.

    :param alphabet: Known alphabet of sequence items.
    :param sequence: sequence of items
    :param wildcard_character: Wildcard character
    :return: Sequence of items where unknowns are replaced with wildcard character."""

    alphabet_lookup = set(alphabet)
    new_sequence = []
    for item in sequence:
        if item in alphabet_lookup:
            new_sequence.append(item)
        else:
            new_sequence.append(wildcard_character)

    return new_sequence


def _pairwise_alignment(
    aligner: PairwiseAligner,
    t: NDArray[np.int32],
    q: NDArray[np.int32],
    gap_repr: np.int32 = np.int32(-1),
    trim: bool = False
) -> tuple[float, NDArray[np.int32], NDArray[np.int32]] | None:
    """
    Align two sequences and return the alignment result.

    :param aligner: PairwiseAligner object to use for alignment.
    :param t: First sequence as a numpy array of integers.
    :param q: Second sequence as a numpy array of integers.
    :param gap_repr: Integer representation of a gap in the alignment (default: -1).
    :return: Tuple containing the alignment score, aligned first sequence, and aligned second sequence.
    :raises ValueError: If the alignment coordinates are invalid.
    """
    alignments = aligner.align(seqA=t, seqB=q)

    if not alignments:
        return None

    # Pick first alignment
    alignment = alignments[0]
    score = alignments[0].score

    t_a: list[np.int32] = []
    q_a: list[np.int32] = []

    if not trim:
        # --- Handle unaligned prefix (relevant for local alignment) ---
        t_start = alignment.coordinates[0][0]
        q_start = alignment.coordinates[1][0]

        if t_start > 0:
            t_a.extend(t[0:t_start])
            q_a.extend([gap_repr] * t_start)
        if q_start > 0:
            t_a.extend([gap_repr] * q_start)
            q_a.extend(q[0:q_start])

    for i in range(alignment.coordinates.shape[1] - 1):
        a = alignment.coordinates[0][i:i + 2]
        b = alignment.coordinates[1][i:i + 2]
        len_a = a[1] - a[0]
        len_b = b[1] - b[0]
        if len_a == len_b:
            t_a.extend(t[a[0]:a[1]])
            q_a.extend(q[b[0]:b[1]])
        elif len_a == 0:
            t_a.extend([gap_repr] * len_b)
            q_a.extend(q[b[0]:b[1]])
        elif len_b == 0:
            t_a.extend(t[a[0]:a[1]])
            q_a.extend([gap_repr] * len_a)
        else:
            raise ValueError("Invalid alignment coordinates!")

    if not trim:
        # --- Handle unaligned suffix (relevant for local alignment) ---
        t_end = alignment.coordinates[0][-1]
        q_end = alignment.coordinates[1][-1]

        t_suffix_len = len(t) - t_end
        q_suffix_len = len(q) - q_end

        if t_suffix_len > 0:
            t_a.extend(t[t_end:])
            q_a.extend([gap_repr] * t_suffix_len)
        if q_suffix_len > 0:
            t_a.extend([gap_repr] * q_suffix_len)
            q_a.extend(q[q_end:])

    return score, np.array(t_a, dtype=np.int32), np.array(q_a, dtype=np.int32)


def align(
    aligner: PairwiseAligner,
    t: list[T],
    q: list[T],
    converter: Converter,
    trim: bool
) -> tuple[float, list[T | None], list[T | None]]:
    """
    Align two sequences and return the alignment result.

    :param aligner: PairwiseAligner object to use for alignment
    :param t: First sequence as a list of items.
    :param q: Second sequence as a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment and back.
    :return: Tuple containing the alignment score, aligned first sequence, and aligned second sequence.
    """
    t_int = np.array([converter.to_identifier(item) for item in t], dtype=np.int32)
    q_int = np.array([converter.to_identifier(item) for item in q], dtype=np.int32)

    s, t_a, q_a = _pairwise_alignment(aligner=aligner, t=t_int, q=q_int, gap_repr=converter.gap_repr, trim=trim)

    t_a_converted = [converter.from_identifier(item) if item != converter.gap_repr else None for item in t_a]
    q_a_converted = [converter.from_identifier(item) if item != converter.gap_repr else None for item in q_a]

    return s, t_a_converted, q_a_converted
