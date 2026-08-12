"""Module for parsing structure variant data for expanding the substitution matrix"""

from dataclasses import dataclass
from enum import Flag, Enum

import pandas as pd

from natu.constants import StructureData


class Modification(Flag):
    """
    Enum for storing substrate modifications
    """
    D = 1
    NME = 2
    D_NME = D | NME

class Chirality(Enum):
    """
    Enum for storing substrate chirality
    """
    L = 1
    D = 2
    X = 3

class NMethylated(Enum):
    """
    Enum for storing substrate N-methylation
    """
    NME = 1
    BASE = 2
    X = 3


@dataclass
class Variant:
    """
    Class for storing structure variant data
    """
    name: str
    smiles: str
    base: Variant | None
    chirality: Chirality
    methylation: NMethylated
    modification: Modification | None


def parse_variants(structure_data: StructureData) -> list[Variant]:
    """
    Parse structure data and return a dictionary of variants

    :param structure_data: StructureData enum, points to packaged natu.structures variants.tsv file
    :return: dictionary of variant name to structure variant instance
    """
    variants: list[Variant] = []

    with structure_data.open() as handle:
        df = pd.read_csv(handle, sep="\t")

    for i, row in df.iterrows():
        base = Variant(name=row["substrate"],
                       smiles=row["smiles"],
                       base=None,
                       chirality=Chirality.X,
                       methylation=NMethylated.X,
                       modification=None)

        variants.append(base)

        d_var = None
        nme_var = None

        if not pd.isna(row["d_substrate"]):
            d_var = Variant(name=row["d_substrate"],
                            smiles=row["d_smiles"],
                            base=base,
                            chirality=Chirality.D,
                            methylation=NMethylated.X,
                            modification=Modification.D)
            base.chirality = Chirality.L
            variants.append(d_var)

        if not pd.isna(row["nme_substrate"]):
            nme_var = Variant(name=
                              row["nme_substrate"],
                              smiles=row["nme_smiles"],
                              base=base,
                              chirality=base.chirality,
                              methylation=NMethylated.NME,
                              modification=Modification.NME)
            base.methylation = NMethylated.BASE
            variants.append(nme_var)

        if not pd.isna(row["nme_d_substrate"]):
            nme_d_var = Variant(name=row["nme_d_substrate"],
                                smiles=row["nme_d_smiles"],
                                base=base,
                                chirality=Chirality.D,
                                methylation=NMethylated.NME,
                                modification=Modification.D_NME)

            # This means the D-variant was N-methylatable, so becomes 'BASE' rather than 'X'
            if d_var is not None:
                d_var.methylation = NMethylated.BASE
            else:
                raise ValueError("Cannot have NMe-D variant without a D-variant. Check variants.tsv")

            assert nme_var is not None

            variants.append(nme_d_var)

    return variants
