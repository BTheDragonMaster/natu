from argparse import ArgumentParser, Namespace
from natu.cli import load_config
from typing import Optional, Any
from natu.variants import StructureCollection, remove_equivalents, compute_variants, parse_smiles
from pathlib import Path

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Write NATU-computed variants to files")
    parser.add_argument("-s", "--smiles", required=True, type=Path, help="SMILES file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output file")
    parser.add_argument("-c", "--configuration_file", required=True, type=Path, help="Configuration file")

    return parser.parse_args()


def get_structure_collections(smiles_file: Path, tailoring_config: dict[str, Any]) -> list[StructureCollection]:
    """Get structure variants from SMILES file

    :param smiles_file: path to SMILES file
    :param tailoring_config: tailoring configuration
    :return: list of structure collections
    """
    structures = parse_smiles(smiles_file)

    structure_collections = []

    for name, smiles in structures.items():
        structure_collections.append(compute_variants(name, smiles, tailoring_config))

    structure_collections = remove_equivalents(structure_collections)
    return structure_collections

def main():
    args = parse_args()
    config = load_config(args.configuration_file)
    tailoring_config = config.get("tailoring", {})
    collections = get_structure_collections(args.smiles, tailoring_config)
    with open(args.output, "w") as f:
        f.write("substrate\tsmiles\td_substrate\td_smiles\tnme_substrate\tnme_smiles\td_nme_substrate\td_nme_smiles\n")
    for collection in collections:
            collection.write_variants(args.output)


if __name__ == "__main__":
    main()