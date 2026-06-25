from argparse import ArgumentParser, Namespace

import pandas as pd


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Print specific element from substitution matrix")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to substitution matrix")
    parser.add_argument("-s1", "--substrate_1", required=True, type=str, help="Name of substrate 1")
    parser.add_argument("-s2", "--substrate_2", required=True, type=str, help="Name of substrate 2")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t", index_col=0)
    score = df.loc[args.substrate_1, args.substrate_2]
    print(f"Score between {args.substrate_1} and {args.substrate_2}: {score}")


if __name__ == "__main__":
    main()
