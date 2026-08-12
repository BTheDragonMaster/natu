from argparse import ArgumentParser, Namespace

from pikachu.fingerprinting.similarity import get_jaccard_matrix
import pandas as pd

from get_substrate_variants import parse_smiles

def parse_arguments() -> Namespace:
    """Parse command line arguments

    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to SMILES file")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path to output matrix")
    args = parser.parse_args()
    return args

def main() -> None:
    args = parse_arguments()
    structure_lookup = parse_smiles(args.input)
    distance_dict = get_jaccard_matrix(structure_lookup)
    df = pd.DataFrame.from_dict(distance_dict)
    df = 1 - df
    df.to_csv(args.output, sep="\t")

if __name__ == "__main__":
    main()
