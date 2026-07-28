"""Command line interface for NATU."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from natu.aligner import setup_aligner, substitution_matrices
from natu.pairwise import Converter
from natu.msa import calculate_msa
from natu.constants import GAP_REPR, AlignmentConfiguration, SubstitutionMatrix
from natu.matrix import create_substitution_matrix
from natu.svg import msa_to_svg


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
        "-s",
        "--substitution-matrix",
        type=parse_substitution_matrix,
        choices=list(SubstitutionMatrix),
        required=True,
        help="Substitution matrix to use for sequence alignment.",
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

    # NATU draw
    draw_parser = subparsers.add_parser("draw", help="Draw an aligned FASTA file as SVG.")
    draw_parser.add_argument(
        "-m", "--msa",
        type=Path,
        required=True,
        help="Path to MSA FASTA file.",
    )
    draw_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to output SVG file.",
    )

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


def load_substitution_matrix(substitution_matrix: SubstitutionMatrix, config: dict[str, Any]) -> substitution_matrices.Array:
    """
    Load a substitution matrix from a file or from the packaged NATU default.

    :param substitution_matrix: Substitution matrix to load.
    :param config: Configuration dictionary.
    :return: Substitution matrix.
    :raises ValueError: If substitution matrix is corrupt.
    """
    with substitution_matrix.open() as handle:
        df = pd.read_csv(handle, sep="\t", index_col=0)

    if list(df.index) != list(df.columns):
        raise ValueError("substitution matrix row names and column names must be identical and in the same order")

    return create_substitution_matrix(df, config)


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
    args = cli()

    if args.command == "align":
        # NATU align

        if not args.fasta.is_file():
            raise FileNotFoundError(f"{args.fasta} does not exist")

        if args.config is not None and not args.config.is_file():
            raise FileNotFoundError(f"{args.config} does not exist")

        config = load_config(args.config, args.substitution_matrix)
        substitution_matrix = load_substitution_matrix(args.substitution_matrix, config)

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

        aligner = setup_aligner(substitution_matrix=substitution_matrix, **aligner_config)

        headers, sequences = zip(*read_monomer_fasta(args.fasta))

        aligned, new_order = calculate_msa(
            aligner=aligner,
            to_align=sequences,
            converter=converter,
            center_star=None,
            progressive=args.progressive
        )
        reorderd_headers = [headers[i] for i in new_order]

        with open(args.output, "w", encoding="utf-8") as handle:
            for header, (score, aligned_sequence)  in zip(reorderd_headers, aligned):
                aligned_sequence_str = "|".join([GAP_REPR if item is None else item for item in aligned_sequence])
                handle.write(f">{header}|alignment_score={score:.3f}\n")
                handle.write(f"{aligned_sequence_str}\n")

    elif args.command == "draw":
        # NATU draw

        if not args.msa.is_file():
            raise FileNotFoundError(f"{args.msa} does not exist")

        records = read_monomer_fasta(args.msa)

        svg_str = msa_to_svg(records)

        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(svg_str)

    else:
        raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
