from argparse import ArgumentParser, Namespace

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem.rdchem import Mol
import pandas as pd


def parse_smiles(smiles_file: str) -> dict[str, Mol]:
    """
    Return dictionary of molecule names to Structure objects

    :param smiles_file: tabular file (with header) with molecule names in column 1 and SMILES in column 2
    :return: dictionary of molecule names to Structure objects
    """
    structure_lookup: dict[str, Mol] = {}
    with open(smiles_file, 'r') as smiles_data:
        smiles_data.readline()
        for line in smiles_data:
            line = line.strip()
            if line:
                name, smiles = line.split('\t')
                structure_lookup[name] = Chem.MolFromSmiles(smiles)
    return structure_lookup


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

if __name__ == "__main__":
    args = parse_arguments()
    name_to_mol = parse_smiles(args.input)
    name_to_name_to_feat: dict[str, dict[str, float]] = {}

    radius = 2
    n_bits = 2048

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=True,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )

    for name_1, mol_1 in name_to_mol.items():
        fp1 = generator.GetFingerprint(mol_1)
        name_to_name_to_feat[name_1] = {}
        for name_2, mol_2 in name_to_mol.items():
            fp2 = generator.GetFingerprint(mol_2)

            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            name_to_name_to_feat[name_1][name_2] = similarity

    df = pd.DataFrame.from_dict(name_to_name_to_feat)
    df.to_csv(args.output, sep="\t")



