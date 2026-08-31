"""Command line interface for NATU."""

import argparse
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from natu.aligner import setup_aligner, substitution_matrices
from natu.pairwise import Converter, replace_unknowns_with_wildcards
from natu.msa import calculate_msa
from natu.search import search
from natu.constants import GAP_REPR, AlignmentConfiguration, SubstitutionMatrix
from natu.scoring import create_substitution_matrix
from natu.svg import msa_to_svg
from natu.matrix import build_substitution_matrix
from natu.network import cluster_sequences, sequence_to_label, write_graphml, write_edge_list, write_clusters
from natu.network_viz import load_network, write_html


log = logging.getLogger(__name__)


try:
    __version__ = version("natu")
except PackageNotFoundError:
    # When running in a source checkout and haven't installed package yet, importlib.metadata might not find "natu" in
    # site-packages. Fallback to a hard-coded default:
    __version__ = "unknown"


def parse_substitution_matrix(value: str) -> SubstitutionMatrix:
    """
    Parse a substitution matrix enum from a command line value.

    Accepts enum names case-insensitively, for example:
    MATCH_MISMATCH, match_mismatch, PARAS_BASED, paras_based.

    :param value: Command line value.
    :return: SubstitutionMatrix enum member.
    :raises argparse.ArgumentTypeError: If value does not match a substitution matrix.
    """
    normalized = value.upper()

    try:
        return SubstitutionMatrix[normalized]
    except KeyError as error:
        valid = ", ".join(matrix.name.lower() for matrix in SubstitutionMatrix)
        raise argparse.ArgumentTypeError(f"invalid substitution matrix {value!r}; choose from: {valid}") from error


def cli() -> argparse.Namespace:
    """
    Command line interface for NATU.

    :return: Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="NATU: NRPS Alignment of Thiotemplated Units",
        add_help=False,
    )

    parser.add_argument(
        "-h", "--help",
        action="help",
        help="Show this help message and exit."
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}", help="Show program's version number and exit."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # NATU align
    align_parser = subparsers.add_parser("align", help="Align monomer sequences from FASTA.")
    align_parser.add_argument(
        "-m",
        "--substitution-matrix",
        type=parse_substitution_matrix,
        choices=list(SubstitutionMatrix),
        required=True,
        help="Substitution matrix to use for sequence alignment.",
    )

    align_parser.add_argument(
        "-s",
        "--smiles",
        type=Path,
        default=None,
        help="Path to SMILES file to use for variant calculation and tailoring-aware scoring."
    )
    align_parser.add_argument(
        "-f", "--fasta",
        type=Path,
        required=True,
        help="Path to sequences to align in FASTA format; monomer names separated by '|' symbols (required)."
    )
    align_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to the output file with aligned sequences (required)."
    )
    align_parser.add_argument(
        "-c", "--config",
        type=Path,
        required=False,
        help="Path to the configuration file (default: None)."
    )
    align_parser.add_argument(
        "-p", "--progressive",
        action="store_true",
        help="If given, do progressive MSA using UPGMA guide tree"
    )

    align_parser.add_argument(
        "-x", "--center_star",
        type=int,
        default=None,
        help="Index of center star sequence for center star alignment. If not given, use sequence with \
        the greatest pairwise similarity to all other sequences."
    )

    # NATU draw
    draw_parser = subparsers.add_parser(
        "draw", help="Draw an aligned FASTA file as SVG, or a similarity network as an interactive HTML page."
    )
    draw_input_group = draw_parser.add_mutually_exclusive_group(required=True)
    draw_input_group.add_argument(
        "-m", "--msa",
        type=Path,
        default=None,
        help="Path to MSA FASTA file. Mutually exclusive with --network.",
    )
    draw_input_group.add_argument(
        "-n", "--network",
        type=Path,
        default=None,
        help="Path to a similarity network GraphML file (as written by \"natu cluster\"). "
             "Mutually exclusive with --msa.",
    )
    draw_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to output file: an SVG file for --msa, or an HTML file for --network.",
    )
    draw_parser.add_argument(
        "-H", "--highlight",
        action="append",
        default=None,
        help="Polymer sequence to highlight in a --network drawing, matched exactly against "
             "a node's pipe-separated monomer sequence (the same string shown as its label, "
             "e.g. 'serine|leucine|glycine'). Only the cluster(s) containing a match are "
             "colored; every other cluster is folded into a single neutral color. Repeat "
             "-H to highlight more than one sequence -- each gets its own color. Combines "
             "with --highlight-fasta if both are given. Only valid together with --network; "
             "ignored/invalid with --msa."
    )
    draw_parser.add_argument(
        "-F", "--highlight-fasta",
        type=Path,
        default=None,
        help="FASTA file of polymer sequences to highlight in a --network drawing (same "
             "monomer format as elsewhere in NATU, e.g. 'serine|leucine|glycine' per "
             "record) -- an alternative to typing sequences out with -H, one per record. "
             "Headers in the file are ignored; only the monomer sequences are used, matched "
             "the same exact way -H is. Combines with -H if both are given. Only valid "
             "together with --network; ignored/invalid with --msa."
    )
    draw_parser.add_argument(
        "-C", "--highlight-contains",
        action="store_true",
        help="Change --highlight/--highlight-fasta matching from \"the node's sequence "
             "equals this exactly\" to \"the node's sequence contains this as a contiguous "
             "run of monomers\" -- e.g. -H 'leucine|glycine' -C would also highlight a node "
             "with the sequence 'serine|leucine|glycine|alanine'. Applies to every -H/-F "
             "value in the same draw call (there is no per-value mix of the two modes). "
             "Only valid together with --network; ignored/invalid with --msa."
    )

    # NATU search

    search_parser = subparsers.add_parser("search", help="Search for similar monomer sequences from FASTA")
    search_parser.add_argument(
        "-q", "--query",
        type=Path,
        required=True,
        help="Path to query FASTA file."
    )
    search_parser.add_argument(
        "-f", "--fasta",
        type=Path,
        required=True,
        help="Path to FASTA file with subject sequences."
    )
    search_parser.add_argument(
        "-m",
        "--substitution-matrix",
        type=parse_substitution_matrix,
        choices=list(SubstitutionMatrix),
        required=True,
        help="Substitution matrix to use for sequence alignment.",
    )
    search_parser.add_argument(
        "-s",
        "--smiles",
        type=Path,
        default=None,
        help="Path to SMILES file to use for variant calculation and tailoring-aware scoring."
    )
    search_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to output file.",
    )
    search_parser.add_argument(
        "-c", "--config",
        type=Path,
        required=False,
        help="Path to the configuration file (default: None)."
    )
    search_parser.add_argument(
        "-t", "--threshold",
        type=float,
        required=False,
        default=0.0,
        help="Bitscore threshold for subject sequence inclusion."
    )
    search_parser.add_argument(
        "-a", "--alignment_mode",
        type=str,
        choices=["global", "local", "glocal"],
        default="global",
        help="Alignment mode. 'glocal' (BiG-SCAPE-style) forces the shorter of each "
             "query/subject pair to align end-to-end while leaving the longer sequence's "
             "non-matching overhang free (unpenalized), reconfigured per pair from the "
             "aligner's own open_end_gap_score/extend_end_gap_score (from the active "
             "config's aligner section). Note this reopens the failure mode that "
             "motivated switching the default to 'global': a short, low-complexity "
             "sequence embedded inside an unrelated longer one can score a high glocal "
             "similarity, since the longer sequence's mismatched overhang is free."
    )

    # NATU cluster

    cluster_parser = subparsers.add_parser(
        "cluster", help="Build a sequence similarity network from a FASTA file and cluster it."
    )
    cluster_parser.add_argument(
        "-f", "--fasta",
        type=Path,
        required=True,
        help="Path to FASTA file with sequences to cluster."
    )
    cluster_parser.add_argument(
        "-m",
        "--substitution-matrix",
        type=parse_substitution_matrix,
        choices=list(SubstitutionMatrix),
        required=True,
        help="Substitution matrix to use for sequence alignment.",
    )
    cluster_parser.add_argument(
        "-s",
        "--smiles",
        type=Path,
        default=None,
        help="Path to SMILES file to use for variant calculation and tailoring-aware scoring."
    )
    cluster_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to output directory (created if it does not exist). Writes network.graphml, "
             "edges.tsv, and clusters.tsv.",
    )
    cluster_parser.add_argument(
        "-c", "--config",
        type=Path,
        required=False,
        help="Path to the configuration file (default: None)."
    )
    cluster_parser.add_argument(
        "-t", "--cutoff",
        type=float,
        required=False,
        default=0.0,
        help="Minimum normalized similarity score (self-score normalized, roughly 0-1) required "
             "to draw an edge between two sequences."
    )
    cluster_parser.add_argument(
        "-a", "--alignment_mode",
        type=str,
        choices=["global", "local", "glocal"],
        default="global",
        help="Alignment mode. 'glocal' (BiG-SCAPE-style) forces the shorter of each pair's "
             "two sequences to align end-to-end while leaving the longer sequence's "
             "non-matching overhang free (unpenalized), reconfigured per pair from the "
             "aligner's own open_end_gap_score/extend_end_gap_score (from the active config's "
             "aligner section). Note this reopens the failure mode that motivated switching "
             "the default to 'global': a short, low-complexity sequence embedded inside an "
             "unrelated longer one can score a high glocal similarity, since the longer "
             "sequence's mismatched overhang is free. --min-length, --min-alignment-length, "
             "and --max-unknown-fraction remain just as relevant under 'glocal' as under "
             "'local'."
    )
    cluster_parser.add_argument(
        "-l", "--min-length",
        type=int,
        required=False,
        default=0,
        help="Minimum sequence length (number of monomers) required to include a sequence in "
             "clustering; shorter sequences are dropped before comparison. Short sequences are "
             "the most common source of spurious high-similarity edges, since similarity is "
             "normalized against the shorter sequence's own self-score -- a 2-monomer sequence "
             "only has to match a 2-monomer fragment of a much longer one to score near 1.0. "
             "Default: 0 (no filtering)."
    )
    cluster_parser.add_argument(
        "-L", "--min-alignment-length",
        type=int,
        required=False,
        default=0,
        help="Minimum number of aligned columns (matches, mismatches, and internal gaps) "
             "required to keep an edge, checked for pairs that already clear --cutoff. Unlike "
             "--min-length, this catches two individually long sequences that only share a "
             "short coincidental motif. Only ever computed for pairs that already passed "
             "--cutoff, so it adds negligible cost on top of a full run. Default: 0 (no "
             "filtering)."
    )
    cluster_parser.add_argument(
        "-u", "--max-unknown-fraction",
        type=float,
        required=False,
        default=1.0,
        help="Maximum fraction of a sequence's monomers that may be the alignment config's "
             "wildcard/unknown character (wildcard.wildcard_character in the active config, "
             "e.g. paras_based.yaml) before the sequence is dropped from clustering. Note this "
             "counts the literal wildcard character, so it only has an effect on sequences "
             "that were actually converted with wildcard.wildcard_for_unknowns -- if that "
             "setting is off in the active config, no monomer will ever equal the wildcard "
             "character and this filter is a no-op. Default: 1.0 (no filtering)."
    )

    # NATU build

    build_parser = subparsers.add_parser("build",
                                         help="Build a substitution matrix from a SMILES or variant file.")

    build_parser.add_argument('-s', "--smiles",
                              required=True,
                              type=Path,
                              help="Input SMILES or variant file."
                              )

    build_parser.add_argument('-m', '--matrix_type',
                              default=SubstitutionMatrix.ECFP,
                              type=parse_substitution_matrix,
                              choices=list(SubstitutionMatrix),
                              help="Type of substitution matrix to build.",
                              )

    build_parser.add_argument('-o', '--output',
                              required=True,
                              type=Path,
                              help="Path to output file with substitution matrix.")

    return parser.parse_args()


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively update a dictionary.

    :param base: Base dictionary.
    :param update: Dictionary with values that should override the base dictionary.
    :return: Updated dictionary.
    """
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value

    return base

def process_sequences(alphabet: Iterable[str], sequences: list[list[str]], config: dict[str, Any]) -> list[list[str]]:
    """
    Process a list of sequences.

    :param alphabet: Known alphabet of sequence items.
    :param sequences: list of sequences to process
    :param config: alignment configuration
    :return: list of processed sequences
    """

    wildcard_config = config.get("wildcard", {})

    if not wildcard_config["wildcard_for_unknowns"]:
        return sequences
    else:
        new_sequences = []

        for sequence in sequences:
            new_sequence = replace_unknowns_with_wildcards(alphabet, sequence, wildcard_config["wildcard_character"])
            new_sequences.append(new_sequence)

        return new_sequences


def filter_by_min_length(
    headers: Iterable[str],
    sequences: list[list[str]],
    min_length: int,
) -> tuple[list[str], list[list[str]]]:
    """
    Drop sequences shorter than a minimum length, along with their headers.

    Very short sequences are the most common source of spurious high-similarity edges in a
    similarity network: because similarity is normalized against the shorter of two
    sequences' own self-score, a short sequence only has to match a small fragment of a much
    longer, unrelated sequence to score near 1.0 -- even though that match carries little to
    no biological meaning. Filtering them out before clustering keeps the network focused on
    comparisons where "similarity" reflects the full extent of both sequences, not a
    coincidental partial match.

    :param headers: Sequence identifiers, in the same order as sequences.
    :param sequences: Sequences to filter.
    :param min_length: Minimum sequence length (number of monomers) required to keep a
        sequence. A value of 0 or less keeps everything.
    :return: Tuple of (filtered headers, filtered sequences), in their original relative order.
    """
    kept_headers: list[str] = []
    kept_sequences: list[list[str]] = []

    for header, sequence in zip(headers, sequences):
        if len(sequence) >= min_length:
            kept_headers.append(header)
            kept_sequences.append(sequence)

    return kept_headers, kept_sequences


def filter_by_max_unknown_fraction(
    headers: Iterable[str],
    sequences: list[list[str]],
    wildcard_character: str | None,
    max_fraction: float,
) -> tuple[list[str], list[list[str]]]:
    """
    Drop sequences whose fraction of wildcard/unknown monomers exceeds a maximum, along with
    their headers.

    This counts the literal wildcard character configured for the active alignment config
    (wildcard.wildcard_character, e.g. "X" in paras_based.yaml). That character only appears in
    a sequence if it was put there by process_sequences()/replace_unknowns_with_wildcards(),
    which only runs when wildcard.wildcard_for_unknowns is True -- if that setting is off, no
    monomer will ever equal the wildcard character and this filter has no effect, regardless of
    how many genuinely unrecognized monomers the sequence contains.

    :param headers: Sequence identifiers, in the same order as sequences.
    :param sequences: Sequences to filter.
    :param wildcard_character: The alignment config's wildcard/unknown character, or None if the
        config doesn't define one. Required (non-None) whenever max_fraction < 1.0.
    :param max_fraction: Maximum allowed fraction (0.0-1.0) of a sequence's monomers that may be
        the wildcard character. A value of 1.0 or more keeps everything.
    :return: Tuple of (filtered headers, filtered sequences), in their original relative order.
    """
    if max_fraction >= 1.0:
        return list(headers), list(sequences)

    if wildcard_character is None:
        raise ValueError(
            "max_unknown_fraction filtering was requested, but the active alignment config "
            "does not define a wildcard.wildcard_character to count as \"unknown\""
        )

    kept_headers: list[str] = []
    kept_sequences: list[list[str]] = []

    for header, sequence in zip(headers, sequences):
        if not sequence:
            kept_headers.append(header)
            kept_sequences.append(sequence)
            continue

        unknown_fraction = sum(1 for monomer in sequence if monomer == wildcard_character) / len(sequence)

        if unknown_fraction <= max_fraction:
            kept_headers.append(header)
            kept_sequences.append(sequence)

    return kept_headers, kept_sequences


def load_config(config_file: Path | None = None, matrix_type: SubstitutionMatrix | None = None) -> dict[str, Any]:
    """
    Load the default NATU config and optionally override it with a user config.

    :param config_file: Optional path to a user-provided YAML config file.
    :param matrix_type: Optional substitution matrix type.
    :return: Configuration dictionary.
    """

    default_config = yaml.safe_load(AlignmentConfiguration.MATCH_MISMATCH.read_text())

    if matrix_type:
        default_config = yaml.safe_load(matrix_type.get_config().read_text())

    if default_config is None:
        default_config = {}

    if config_file is None:
        return default_config

    with config_file.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle)

    if user_config is None:
        user_config = {}

    return deep_update(default_config, user_config)


def load_substitution_matrix(substitution_matrix: SubstitutionMatrix, config: dict[str, Any], smiles_file: Path | None = None) -> substitution_matrices.Array:
    """
    Load a substitution matrix from a file or from the packaged NATU default.

    :param substitution_matrix: Substitution matrix to load.
    :param config: Configuration dictionary.
    :param smiles_file: Optional path to SMILES file; necessary for tailoring-aware scoring
    :return: Substitution matrix.
    :raises ValueError: If substitution matrix is corrupt.
    """
    with substitution_matrix.open() as handle:
        df = pd.read_csv(handle, sep="\t", index_col=0)

    if list(df.index) != list(df.columns):
        raise ValueError("substitution matrix row names and column names must be identical and in the same order")

    return create_substitution_matrix(df, config, smiles_file)


def read_monomer_fasta(fasta_file: Path) -> list[tuple[str, list[str]]]:
    """
    Read monomer sequences from a FASTA file while preserving full headers.

    :param fasta_file: Path to FASTA file.
    :return: List of tuples containing record header and monomer sequence.
    :raises ValueError: If the FASTA file is empty or contains invalid records.
    """
    records = []
    header = None
    sequence_lines = []

    def add_record() -> None:
        if header is None:
            return

        sequence = "".join(sequence_lines).strip()
        if not sequence:
            raise ValueError(f"record {header!r} has an empty sequence")

        monomers = sequence.split("|")
        if any(not monomer for monomer in monomers):
            raise ValueError(f"record {header!r} contains an empty monomer")

        records.append((header, monomers))

    with fasta_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                add_record()
                header = line[1:].strip()
                sequence_lines = []
            else:
                if header is None:
                    raise ValueError("found sequence before first FASTA header")
                sequence_lines.append(line)

    add_record()

    if not records:
        raise ValueError(f"no FASTA records found in {fasta_file}")

    return records


def main() -> None:
    """
    Entry point for NATU.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = cli()

    if args.command in ["align", "search", "cluster"]:
        # NATU align and search

        if args.command == "search":
            if not args.query.is_file():
                raise FileNotFoundError(f"{args.query} does not exist")

        if not args.fasta.is_file():
            raise FileNotFoundError(f"{args.fasta} does not exist")

        if args.config is not None and not args.config.is_file():
            raise FileNotFoundError(f"{args.config} does not exist")

        config = load_config(args.config, args.substitution_matrix)
        substitution_matrix = load_substitution_matrix(args.substitution_matrix, config, args.smiles)

        alphabet = tuple(substitution_matrix.alphabet)
        alphabet_to_index = {symbol: np.int32(i) for i, symbol in enumerate(alphabet)}

        def to_identifier(symbol: str) -> np.int32:
            """
            Convert a sequence symbol to its substitution-matrix index.

            :param symbol: Sequence symbol.
            :return: Integer index in the substitution matrix alphabet.
            :raises ValueError: If the symbol is not present in the substitution matrix alphabet.
            """
            try:
                return alphabet_to_index[symbol]
            except KeyError as error:
                raise ValueError(f"symbol {symbol!r} was not found in the substitution matrix alphabet") from error

        def from_identifier(index: np.int32) -> str:
            """
            Convert a substitution-matrix index back to a sequence symbol.

            :param index: Substitution-matrix index.
            :return: Sequence symbol.
            :raises ValueError: If the index is outside the alphabet range.
            """
            try:
                return alphabet[index]
            except IndexError as error:
                raise ValueError(f"index {index!r} is outside the substitution matrix alphabet range") from error

        converter = Converter(to_identifier=to_identifier, from_identifier=from_identifier)

        aligner_config = config.get("aligner", {})

        if args.command == "align":

            aligner = setup_aligner(substitution_matrix=substitution_matrix, **aligner_config)

            headers, sequences = zip(*read_monomer_fasta(args.fasta))
            sequences = process_sequences(alphabet, sequences, config)

            aligned, new_order = calculate_msa(
                aligner=aligner,
                to_align=sequences,
                converter=converter,
                center_star=args.center_star,
                progressive=args.progressive
            )
            reordered_headers = [headers[i] for i in new_order]

            with open(args.output, "w", encoding="utf-8") as handle:
                for header, (score, aligned_sequence)  in zip(reordered_headers, aligned):
                    aligned_sequence_str = "|".join([GAP_REPR if item is None else item for item in aligned_sequence])
                    handle.write(f">{header}|alignment_score={score:.3f}\n")
                    handle.write(f"{aligned_sequence_str}\n")
        elif args.command == "search":

            aligner = setup_aligner(substitution_matrix=substitution_matrix, mode=args.alignment_mode,
                                    **aligner_config)

            query_headers, query_sequences = zip(*read_monomer_fasta(args.query))
            query_sequences = process_sequences(alphabet, query_sequences, config)

            subject_headers, subject_sequences = zip(*read_monomer_fasta(args.fasta))
            subject_sequences = process_sequences(alphabet, subject_sequences, config)

            search_results = search(
                aligner=aligner,
                query_sequences=query_sequences,
                subject_sequences=subject_sequences,
                converter=converter,
                threshold=args.threshold,
                trim=config.get("trim", False),
                glocal=args.alignment_mode == "glocal")

            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write("query\tsubject\tbitscore\taligned_q\ts_aligned_s\n")

                for i, results in enumerate(search_results):
                    query_header = query_headers[i]
                    for result in results:
                        score, gapped_query, gapped_subject, s_idx = result
                        aligned_query_str = "|".join(
                            [GAP_REPR if item is None else item for item in gapped_query])
                        aligned_subject_str = "|".join(
                            [GAP_REPR if item is None else item for item in gapped_subject])
                        subject_header = subject_headers[s_idx]
                        handle.write(f"{query_header}\t{subject_header}\t{score:.3f}\t{aligned_query_str}\t{aligned_subject_str}\n")

        elif args.command == "cluster":

            aligner = setup_aligner(substitution_matrix=substitution_matrix, mode=args.alignment_mode,
                                    **aligner_config)

            headers, sequences = zip(*read_monomer_fasta(args.fasta))
            sequences = process_sequences(alphabet, sequences, config)

            if args.min_length > 0:
                n_before = len(headers)
                headers, sequences = filter_by_min_length(headers, sequences, args.min_length)
                log.info(
                    "cluster: kept %d/%d sequences at length >= %d monomers (dropped %d)",
                    len(headers), n_before, args.min_length, n_before - len(headers),
                )

            if args.max_unknown_fraction < 1.0:
                n_before = len(headers)
                wildcard_character = config.get("wildcard", {}).get("wildcard_character")
                headers, sequences = filter_by_max_unknown_fraction(
                    headers, sequences, wildcard_character, args.max_unknown_fraction
                )
                log.info(
                    "cluster: kept %d/%d sequences at <= %.0f%% unknown ('%s') monomers (dropped %d)",
                    len(headers), n_before, args.max_unknown_fraction * 100, wildcard_character,
                    n_before - len(headers),
                )

            graph, clusters = cluster_sequences(
                aligner=aligner,
                headers=list(headers),
                sequences=list(sequences),
                converter=converter,
                cutoff=args.cutoff,
                min_alignment_length=args.min_alignment_length,
                glocal=args.alignment_mode == "glocal",
            )

            args.output.mkdir(parents=True, exist_ok=True)
            write_graphml(graph, args.output / "network.graphml")
            write_edge_list(graph, args.output / "edges.tsv")
            write_clusters(clusters, args.output / "clusters.tsv")

            print(
                f"{len(headers)} sequences, {graph.number_of_edges()} edges >= cutoff, "
                f"{len(clusters)} clusters (largest: {len(clusters[0]) if clusters else 0})"
            )

        else:
            raise ValueError(f"Unknown command {args.command}")


    elif args.command == "draw":
        # NATU draw

        if args.network is not None:
            if not args.network.is_file():
                raise FileNotFoundError(f"{args.network} does not exist")

            highlight = list(args.highlight) if args.highlight else []
            # No name to show for a raw -H/--highlight value -- it's only ever a sequence
            # typed on the command line, so its own legend row falls back to showing that
            # sequence (see natu.network_viz._assign_highlight_slots).
            highlight_names: list[str | None] = [None] * len(highlight)

            if args.highlight_fasta is not None:
                if not args.highlight_fasta.is_file():
                    raise FileNotFoundError(f"{args.highlight_fasta} does not exist")

                # Sequences are turned into the same pipe-joined string a network node's
                # "sequence" attribute uses, so matching behaves identically whether a
                # sequence came from -H or from this file. Headers are kept alongside
                # (not used for matching) purely so the legend can show a readable name
                # instead of the full sequence for each --highlight-fasta entry.
                highlight_records = read_monomer_fasta(args.highlight_fasta)
                for header, sequence in highlight_records:
                    highlight.append(sequence_to_label(sequence))
                    highlight_names.append(header)

            graph = load_network(args.network)
            write_html(
                graph,
                args.output,
                highlight=highlight or None,
                highlight_contains=args.highlight_contains,
                highlight_names=highlight_names or None,
            )
        else:
            if args.highlight:
                raise ValueError("--highlight is only valid together with --network, not --msa")

            if args.highlight_fasta:
                raise ValueError("--highlight-fasta is only valid together with --network, not --msa")

            if args.highlight_contains:
                raise ValueError("--highlight-contains is only valid together with --network, not --msa")

            if not args.msa.is_file():
                raise FileNotFoundError(f"{args.msa} does not exist")

            records = read_monomer_fasta(args.msa)

            svg_str = msa_to_svg(records)

            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(svg_str)
    elif args.command == "build":
        build_substitution_matrix(args.smiles, args.matrix_type, args.output)
    else:
        raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
