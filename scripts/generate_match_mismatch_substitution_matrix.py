"""Script for creating a match-mismatch substitution matrix for PARAS substrates."""

import argparse
from pathlib import Path

import pandas as pd
from parasect.core.constants import SMILES_FILE


def cli() -> argparse.Namespace:
    """
    Command line interface.

    :return: Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path for the tab-separated match-mismatch substitution matrix for PARAS-based substrates."
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for script.
    """
    args = cli()

    df = pd.read_csv(SMILES_FILE, sep="\t")
    substrate_names = df["substrate"].unique().tolist()

    matrix = pd.DataFrame(
        0,
        index=substrate_names,
        columns=substrate_names,
        dtype=float,
    )

    for substrate in substrate_names:
        matrix.loc[substrate, substrate] = 1.0

    matrix.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()
