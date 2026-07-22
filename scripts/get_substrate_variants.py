from argparse import ArgumentParser, Namespace
import os
from dataclasses import dataclass
from enum import Flag

from pikachu.reactions.functional_groups import GroupDefiner, find_atoms, combine_structures
from pikachu.chem.structure import Structure
from pikachu.general import read_smiles
from pikachu.general import structure_to_smiles

C1_L_AMINO_ACID = GroupDefiner("C1 L-amino acid", "N[C@@H](C)C(=O)O", 1)
C1_D_AMINO_ACID = GroupDefiner("C1 D-amino acid", "N[C@H](C)C(=O)O", 1)
N_AMINO_ACID = GroupDefiner("Nitrogen atom amino acid", "NCC(=O)O", 0)
B_N_AMINO_ACID = GroupDefiner("Nitrogen atom beta amino acid", "NCCC(=O)O", 0) # Example: branched aspartic acid
G_N_AMINO_ACID = GroupDefiner("Nitrogen atom beta amino acid", "NCCCC(=O)O", 0) # Example: branched glutamic acid


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-s", "--smiles_file", required=True, help="SMILES file")
    parser.add_argument("-o", "--output_directory", required=True, help="Output directory")
    parser.add_argument("-p", "--paras_results", required=True,
                        help="PARAS results file predicting all substrates, used as reference")
    args = parser.parse_args()
    return args

#Taken from RAIChU
def epimerization(chiral_centre):
    new_chirality = None
    if chiral_centre.chiral == 'clockwise':
        new_chirality = 'counterclockwise'
    elif chiral_centre.chiral == 'counterclockwise':
        new_chirality = 'clockwise'

    chiral_centre.chiral = new_chirality

# Taken from RAIChU
def methylation(target_atom, structure):

    methyl_group = read_smiles('C')
    carbon = methyl_group.atoms[0]
    hydrogen_1 = methyl_group.atoms[1]
    bond_1 = carbon.get_bond(hydrogen_1)

    hydrogen_2 = target_atom.get_neighbour('H')

    if not hydrogen_2:
        raise Exception("Can't methylate this atom!")

    bond_2 = target_atom.get_bond(hydrogen_2)

    methylated_structure = combine_structures([structure, methyl_group])

    methylated_structure.break_bond(bond_1)
    methylated_structure.break_bond(bond_2)

    methylated_structure.make_bond(carbon, target_atom, methylated_structure.find_next_bond_nr())
    methylated_structure.make_bond(hydrogen_1, hydrogen_2, methylated_structure.find_next_bond_nr())

    structures = methylated_structure.split_disconnected_structures()

    for s in structures:
        if carbon.nr in s.atoms:
            return s

    else:
        raise ValueError("Cannot methylate this atom!")


def parse_smiles(smiles_file: str) -> dict[str, Structure]:
    structure_lookup: dict[str, Structure] = {}
    with open(smiles_file, 'r') as smiles_data:
        smiles_data.readline()
        for line in smiles_data:
            line = line.strip()
            if line:
                name, smiles = line.split('\t')
                structure_lookup[name] = read_smiles(smiles)
    return structure_lookup

def get_d_structure(structure: Structure) -> Structure | None:
    structure = structure.deepcopy()
    c1_atoms_aa = find_atoms(C1_L_AMINO_ACID, structure)

    if not c1_atoms_aa:
        return None
    elif len(c1_atoms_aa) == 1:
        epimerization(c1_atoms_aa[0])
        return structure
    else:
        print(f"More than one chiral centre found")
        return None


def get_nme_structure(name: str, structure: Structure) -> Structure | None:
    structure = structure.deepcopy()

    if "branched" not in name:

        n_atoms_aa = find_atoms(N_AMINO_ACID, structure)
        if not n_atoms_aa:
            n_atoms_aa = find_atoms(B_N_AMINO_ACID, structure)
    else:
        n_atoms_aa = find_atoms(B_N_AMINO_ACID, structure)
        if not n_atoms_aa:
            n_atoms_aa = find_atoms(G_N_AMINO_ACID, structure)

    if not n_atoms_aa:
        return None

    elif len(n_atoms_aa) == 1:

        n_atom = n_atoms_aa[0]
        if not n_atom.has_neighbour('H'):
            return None
        else:
            structure = methylation(n_atom, structure)
            return structure
    else:
        print("More than one methylation site found")
        return None


def is_equivalent(structure_1: Structure, structure_2: Structure) -> bool:
    if structure_1.find_substructures(structure_2) and structure_2.find_substructures(structure_1):
        return True
    return False

def remove_equivalents(structure_collections: list[StructureCollection]) -> list[StructureCollection]:
    base_structures = [c.base_structure for c in structure_collections]
    d_variants = [d for c in structure_collections for d in c.variants if d.modifications == Modification.D]

    structures_to_remove: list[StructureVariant] = []
    for base_structure in base_structures:
        for d_variant in d_variants:
            if is_equivalent(base_structure.structure, d_variant.structure):
                if ("branched" in base_structure.name and not "branched" in d_variant.name) or \
                        ("branched" in d_variant.name and not "branched" in base_structure.name):
                    continue
                structures_to_remove.append(d_variant)

    filtered_structures: list[StructureCollection] = []

    for collection in structure_collections:
        if collection.base_structure not in structures_to_remove:
            filtered_structures.append(collection)

    return filtered_structures

def parse_paras_substrates(paras_file: str) -> list[str]:
    with open(paras_file, 'r') as paras_data:
        paras_data.readline()
        substrate_line = paras_data.readline()
        substrate_line.strip()
        substrates = []
        substrates_and_conf = substrate_line.split('\t')[1:]
        for i, substrate in enumerate(substrates_and_conf):
            if i % 2 == 0:
                substrates.append(substrate)

    return substrates

class Modification(Flag):
    D = 1
    NME = 2
    ME = 4

class StructureCollection:
    def __init__(self, base_structure: StructureVariant, variants: list[StructureVariant]):
        self.base_structure = base_structure
        self.variants = variants

    def get_variant_from_modification(self, modification: Modification) -> StructureVariant | None:
        for variant in self.variants:
            if variant.modifications == modification:
                return variant
        return None

    def write_variants(self, file_name: str) -> None:
        with open(file_name, "a") as f:
            variant_strings: list[str] = []
            f.write(f"{self.base_structure.name}\t{self.base_structure.smiles}\t")
            for modification in [Modification.D, Modification.NME, Modification.D | Modification.NME]:
                variant = self.get_variant_from_modification(modification)
                if variant is None:
                    variant_strings.append("\t")
                else:
                    variant_strings.append(f"{variant.name}\t{variant.smiles}")

            f.write(f"{'\t'.join(variant_strings)}\n")

    def write_smiles(self, file_name: str) -> None:
        with open(file_name, "a") as f:
            f.write(f"{self.base_structure.name}\t{self.base_structure.smiles}\n")
            for variant in self.variants:
                f.write(f"{variant.name}\t{variant.smiles}\n")

@dataclass
class StructureVariant:

    name: str
    modifications: Modification | None
    structure: Structure

    def __post_init__(self):
        self.smiles = structure_to_smiles(self.structure)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

def get_structure_variants(name: str, structure: Structure) -> StructureCollection:
    d_structure = get_d_structure(structure)
    nme_structure = get_nme_structure(name, structure)
    nme_d_structure = None

    base = StructureVariant(name=name, modifications=None, structure=structure)

    variants = []

    if nme_structure is not None:

        nme_name = f"NMe-{name}"
        variants.append(StructureVariant(name=nme_name, modifications=Modification.NME, structure=nme_structure))

    if d_structure is not None:
        if "2S" in name:
            d_name = name.replace("2S", "2R")
        elif name in ["isoleucine", "threonine"]:
            d_name = f"D-allo-{name}"
        elif name in ["allo-isoleucine", 'allo-threonine']:
            d_name = name.replace("allo-", "D-")
        else:
            d_name = f"D-{name}"

        variants.append(StructureVariant(name=d_name, modifications=Modification.D, structure=d_structure))
        nme_d_structure = get_nme_structure(name, d_structure)

    if nme_d_structure is not None:
        if "2S" in name:
            nme_d_name = f"NMe-{name.replace('2S', '2R')}"
        elif name in ["isoleucine", "threonine"]:
            nme_d_name = f"NMe-D-allo-{name}"
        elif name in ["allo-isoleucine", 'allo-threonine']:
            nme_d_name = name.replace("allo-", "NMe-D-")
        else:
            nme_d_name = f"NMe-D-{name}"

        modifications = Modification.NME | Modification.D

        variants.append(StructureVariant(name=nme_d_name, modifications=modifications, structure=nme_d_structure))

    structure_collection = StructureCollection(base, variants)
    return structure_collection


def main():
    args = parse_args()
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    structures = parse_smiles(args.smiles_file)
    paras_substrates = parse_paras_substrates(args.paras_results)

    paras_structures: dict[str, Structure] = {}

    for name in paras_substrates:
        paras_structures[name] = structures[name]

    structure_collections = []


    for name, structure in paras_structures.items():
        structure_collections.append(get_structure_variants(name, structure))

    structure_collections = remove_equivalents(structure_collections)


    variant_output = os.path.join(args.output_directory, "variants.tsv")
    smiles_output = os.path.join(args.output_directory, "smiles.tsv")

    with open(variant_output, "w") as out:
        out.write(f"substrate\tsmiles\td_substrate\td_smiles\tnme_substrate\tnme_smiles\tnme_d_substrate\tnme_d_smiles\n")

    for collection in structure_collections:
        collection.write_variants(variant_output)
        collection.write_smiles(smiles_output)


if __name__ == "__main__":
    main()
