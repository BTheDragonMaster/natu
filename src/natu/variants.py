"""Module for calculating structure variant data for expanding the substitution matrix"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from typing import Any, Optional
import logging

from pikachu.reactions.functional_groups import GroupDefiner, find_atoms, combine_structures
from pikachu.chem.structure import Structure
from pikachu.chem.atom import Atom
from pikachu.general import read_smiles
from pikachu.general import structure_to_smiles

# Determines L and D stereochemistry
ALPHA_CARBON_L_AMINO_ACID = GroupDefiner("alpha-carbon L-amino acid", "N[C@@H](C)C(=O)O", 1)
ALPHA_CARBON_D_AMINO_ACID = GroupDefiner("alpha-carbon D-amino acid", "N[C@H](C)C(=O)O", 1)

# Used to determine if beta-carbon is chiral
BETA_CARBON = GroupDefiner("Beta-carbon", "NC(C)C(=O)O", 2)

N_AMINO_ACID = GroupDefiner("Nitrogen atom amino acid", "NCC(=O)O", 0)

B_N_AMINO_ACID = GroupDefiner("Nitrogen atom beta amino acid", "NCCC(=O)O", 0) # Example: branched aspartic acid
G_N_AMINO_ACID = GroupDefiner("Nitrogen atom beta amino acid", "NCCCC(=O)O", 0) # Example: branched glutamic acid

logger = logging.getLogger(__name__)

class Modification(Enum):
    """
    Enum for storing substrate modifications
    """
    chirality = 1
    n_methylation = 2


class Chirality(Enum):
    """
    Enum for storing substrate chirality
    """
    L = 1
    D = 2
    X = 3


class NMethylation(Enum):
    """
    Enum for storing substrate N-methylation
    """
    Y = 1
    N = 2
    X = 3


@dataclass
class Variant:
    """
    Class for storing structure variant data
    """
    name: str
    smiles: str
    structure: Structure
    base: "Variant | None"
    chirality: Chirality
    n_methylation: NMethylation
    modifications: list[Modification]


#Taken from RAIChU
def epimerization(chiral_centre: Atom) -> None:
    """
    Change chirality at given atom

    :param chiral_centre: PIKAChU Atom object
    """
    new_chirality = None
    if chiral_centre.chiral == 'clockwise':
        new_chirality = 'counterclockwise'
    elif chiral_centre.chiral == 'counterclockwise':
        new_chirality = 'clockwise'

    chiral_centre.chiral = new_chirality

# Taken from RAIChU
def methylation(target_atom, structure) -> Structure:
    """
    Methylate structure at given atom

    :param target_atom: PIKAChU Atom object
    :param structure: PIKAChU Structure object
    :return: methylated structure
    """

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

def parse_smiles(smiles_file: Path) -> dict[str, str]:
    """
    Return dictionary of molecule names to Structure objects

    :param smiles_file: tabular file (with header) with molecule names in column 1 and SMILES in column 2
    :return: dictionary of molecule names to Structure objects
    """
    smiles_lookup: dict[str, str] = {}
    with smiles_file.open('r') as smiles_data:
        smiles_data.readline()
        for line in smiles_data:
            line = line.strip()
            if line:
                name, smiles = line.split('\t')
                smiles_lookup[name] = smiles
    return smiles_lookup

def get_d_structure(structure: Structure) -> Structure | None:
    """
    Return D-enantiomer if structure is L-amino acid; None otherwise

    :param structure: PIKAChU Structure object
    :return: PIKAChU Structure object of D-enantiomer if structure is L-amino acid, None otherwise
    """
    structure = structure.deepcopy()
    c1_atoms_aa = find_atoms(ALPHA_CARBON_L_AMINO_ACID, structure)

    if not c1_atoms_aa:
        return None
    elif len(c1_atoms_aa) == 1:
        epimerization(c1_atoms_aa[0])
        return structure
    else:
        logger.debug(f"More than one chiral centre found")
        return None


def get_nme_structure(structure: Structure) -> Structure | None:
    """Return N-methylated structure if structure is amino acid and can be methylated at N; None otherwise

    :param structure: PIKAChU Structure object
    :return: PIKAChU Structure object of N-methylated if structure is amino acid, None otherwise
    """
    structure = structure.deepcopy()

    n_atoms_aa = find_atoms(N_AMINO_ACID, structure)
    if not n_atoms_aa:
        n_atoms_aa = find_atoms(B_N_AMINO_ACID, structure)

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
        logger.debug("More than one methylation site found")
        return None


def is_equivalent(structure_1: Structure, structure_2: Structure) -> bool:
    """ Return True if two structures are equivalent, False otherwise

    :param structure_1: PIKAChU Structure object
    :param structure_2: PIKAChU Structure object
    :return: True if two structures are equivalent, False otherwise"""
    if structure_1.find_substructures(structure_2) and structure_2.find_substructures(structure_1):
        return True
    return False

def substrates_from_matrix(matrix_file: str) -> list[str]:
    """Get substrate names from a substitution matrix file

    :param matrix_file: Path to substitution matrix file (see src.natu.data.substitution_matrices)
    :return: list of substrate names
    """
    with open(matrix_file, 'r') as matrix:
        header = matrix.readline()
        header = header.strip()
        substrates = header.split('\t')
        return substrates

class StructureCollection:
    """
    Class to store structure variants for a substrate
    """
    def __init__(self, base_structure: Variant, variants: list[Variant]):
        self.base_structure = base_structure
        self.variants = variants

    def get_variant_from_modifications(self, modifications: list[Modification]) -> Variant | None:
        for variant in self.variants:
            if set(variant.modifications) == set(modifications):
                return variant
        return None

    def write_variants(self, file_name: str) -> None:
        with open(file_name, "a") as f:
            variant_strings: list[str] = []
            f.write(f"{self.base_structure.name}\t{self.base_structure.smiles}\t")
            for modification in [[Modification.chirality], [Modification.n_methylation], [Modification.chirality, Modification.n_methylation]]:
                variant = self.get_variant_from_modifications(modification)
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

def modify(name: str, exclusions: list[str]) -> bool:
    """Determine if a substrate is excluded from modification by the user

    :param name: name of the substrate
    :param exclusions: list of substrings that mark a substrate as excluded
    :return:
    """
    modify_substrate = True
    for exclusion in exclusions:
        if exclusion in name:
            modify_substrate = False
            break
    return modify_substrate

def has_chiral_beta_carbon(structure: Structure) -> bool:
    """Return True if the beta-carbon of the structure is chiral, False otherwise

    :param structure: PIKAChU Structure object
    :return: True if the beta-carbon of the structure is chiral, False otherwise
    """

    has_chiral = False

    structure = structure.deepcopy()
    beta_carbons = find_atoms(BETA_CARBON, structure)

    if beta_carbons:
        beta_carbon = beta_carbons[0]
        if beta_carbon.chiral:
            has_chiral = True

    return has_chiral


def get_variant_name(name: str,
                     structure: Structure,
                     tailoring_config: dict[str, Any],
                     modifications: list[Modification]) -> str:
    """Return a variant name based on a list of modifications on the base structure

    :param name: base name of the structure variant
    :param structure: PIKAChU Structure object of the variant
    :param tailoring_config: tailoring configuration
    :param modifications: list of modifications of the variant compared to the base structure
    :return: name of the variant
    """
    if Modification.chirality in modifications:
        chirality_config = tailoring_config.get("chirality", {})
        chirality_prefixes = chirality_config["nomenclature"]


        contained_chirality_prefixes = []
        for prefix in chirality_prefixes:
            if prefix is not None and prefix in name:
                contained_chirality_prefixes.append(prefix)
        if len(contained_chirality_prefixes) > 1:
            logger.debug(
                "More than one chirality prefix found in name. Choosing longest. Chirality prefix options can be found in the alignment configuration.yaml file.")
            existing_chirality_prefix = max(contained_chirality_prefixes, key=len)
        elif len(contained_chirality_prefixes) == 1:
            existing_chirality_prefix = contained_chirality_prefixes[0]
        else:
            existing_chirality_prefix = None

        allo_prefix = ""
        add_allo_prefix = True
        for excluded_substring in chirality_config["allo_exclusions"]:
            if excluded_substring in name:
                add_allo_prefix = False

        if has_chiral_beta_carbon(structure) and add_allo_prefix:
            if chirality_config["allo_prefix"] in name:
                name = name.replace(chirality_config["allo_prefix"], '')
            else:
                allo_prefix = chirality_config["allo_prefix"]

        if existing_chirality_prefix is not None:
            name = allo_prefix + name.replace(existing_chirality_prefix, chirality_prefixes[existing_chirality_prefix])
        else:
            name = chirality_prefixes[existing_chirality_prefix] + allo_prefix + name


    if Modification.n_methylation in modifications:
        n_methylation_config = tailoring_config.get("n_methylation", {})
        n_methylation_prefix = n_methylation_config["prefix"]
        name = n_methylation_prefix + name

    return name


def compute_variants(name: str, smiles: str, tailoring_config: dict[str, Any]) -> StructureCollection:
    """Return collection of structure variants for a substrate

    :param name: name of the base structure
    :param smiles: SMILES string of the base structure
    :param tailoring_config: Tailoring configuration

    :return: structure collection
    """
    chirality_config = tailoring_config.get("chirality", {})
    n_methylation_config = tailoring_config.get("n_methylation", {})

    chirality_exclusions = chirality_config["exclude"]
    n_methylation_exclusions = n_methylation_config["exclude"]

    structure = read_smiles(smiles)
    d_structure = None
    nme_structure = None
    nme_d_structure = None

    if chirality_config["chirality_aware_scoring"]:
        d_structure = get_d_structure(structure)

    if n_methylation_config["n_methylation_aware_scoring"]:
        nme_structure = get_nme_structure(structure)

    base = Variant(name=name,
                   smiles=smiles,
                   structure=structure,
                   base=None,
                   chirality=Chirality.X,
                   n_methylation=NMethylation.X, modifications=[])

    variants = []

    if nme_structure is not None and modify(name, n_methylation_exclusions):

        base.n_methylation = NMethylation.N
        modifications = [Modification.n_methylation]

        nme_name = get_variant_name(name, nme_structure, tailoring_config, modifications)

        variant = Variant(name=nme_name,
                          smiles=structure_to_smiles(nme_structure),
                          structure=nme_structure,
                          base=base,
                          chirality=Chirality.X,
                          n_methylation=NMethylation.Y,
                          modifications=[Modification.n_methylation]
                          )
        variants.append(variant)

    if d_structure is not None and modify(name, chirality_exclusions):
        base.chirality = Chirality.L
        modifications = [Modification.chirality]

        d_name = get_variant_name(name, d_structure, tailoring_config, modifications)

        variant = Variant(name=d_name,
                          smiles=structure_to_smiles(d_structure),
                          structure=d_structure,
                          base=base,
                          chirality=Chirality.D,
                          n_methylation=base.n_methylation,
                          modifications=[Modification.chirality])

        variants.append(variant)
        if nme_structure is not None:
            nme_d_structure = get_nme_structure(d_structure)

    if nme_d_structure is not None:
        modifications = [Modification.chirality, Modification.n_methylation]
        nme_d_name = get_variant_name(name, nme_d_structure, tailoring_config, modifications)

        variant = Variant(name=nme_d_name,
                          smiles=structure_to_smiles(nme_d_structure),
                          structure=nme_d_structure,
                          base=base,
                          chirality=Chirality.D,
                          n_methylation=NMethylation.Y,
                          modifications=[Modification.chirality, Modification.n_methylation])
        variants.append(variant)

    structure_collection = StructureCollection(base, variants)

    return structure_collection


def remove_equivalents(structure_collections: list[StructureCollection]) -> list[StructureCollection]:
    """Detect equivalent structures

    :param structure_collections: list of structure collections
    :return: list of structure collections with duplicate structures removed
    """

    structures_to_remove: list[str] = []
    for i, collection_1 in enumerate(structure_collections):
        base_structure_1 = collection_1.base_structure.structure
        for j, collection_2 in enumerate(structure_collections[i + 1:]):
            base_structure_2 = collection_2.base_structure.structure
            if is_equivalent(base_structure_1, base_structure_2):
                logger.debug(
                    f"WARNING: Monomers {collection_1.base_structure.name} and {collection_2.base_structure.name} in SMILES list are structurally identical. If you do not want to keep both, please remove one of these from your SMILES file")
            if collection_1.base_structure.name == collection_2.base_structure.name:
                raise ValueError(f"Duplicate substrates found in SMILES input file: {collection_1.base_structure.name}. Please remove all duplicates.")

            for variant in collection_2.variants:
                if is_equivalent(base_structure_1, variant.structure):
                    if collection_1.base_structure.name == variant.name:
                        logger.debug(
                            f"WARNING: Structure variant of {variant.base.name} found in SMILES input file: {variant.name}. Substrate {variant.name} is treated as variant of {variant.base.name}.")
                        structures_to_remove.append(variant.name)
                    else:
                        logger.debug(f"WARNING: Structure variant of {variant.base.name}, {collection_1.base_structure.name} found in SMILES input file with mismatching name: {variant.name}. If you do not want to keep both, please remove one of these from your SMILES file")

    filtered_structures: list[StructureCollection] = []

    for collection in structure_collections:
        if collection.base_structure.name not in structures_to_remove:
            filtered_structures.append(collection)

    return filtered_structures


def get_structure_variants(smiles_file: Path, tailoring_config: dict[str, Any], substrates: Optional[list[str]] = None) -> list[Variant]:
    """Get structure variants from SMILES file

    :param smiles_file: path to SMILES file
    :param tailoring_config: tailoring configuration
    :param substrates: list of substrates to include. If None, compute variants for all substrates
    :return: list of structure variants
    """
    structures = parse_smiles(smiles_file)
    if substrates is not None:
        filtered_structures = {}
        for substrate in substrates:
            filtered_structures[substrate] = structures[substrate]

        structures = filtered_structures

    structure_collections = []

    for name, smiles in structures.items():
        structure_collections.append(compute_variants(name, smiles, tailoring_config))

    structure_collections = remove_equivalents(structure_collections)
    variants: list[Variant] = []
    for collection in structure_collections:
        variants.append(collection.base_structure)
        for variant in collection.variants:
            variants.append(variant)

    return variants
