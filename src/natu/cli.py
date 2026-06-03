"""
Command line interface for NATU.
"""

import argparse


def cli() -> argparse.Namespace:
    """
    Command line interface for NATU.

    :return: Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main() -> None:
    """
    Entry point for NATU.
    """
    args = cli()


if __name__ == "__main__":
    main()
