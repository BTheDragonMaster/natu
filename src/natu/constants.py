"""Module for retrieving configs and substitution matrices shipped with NATU."""

from enum import Enum
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import IO


GAP_REPR = "GAP"

class StructureData(Enum):
    """
    Packages variant data files.

    Each enum member points to a TSV file containing the variant data
    """
    DEFAULT = "smiles.tsv"
    PARAS = "paras_smiles.tsv"

    @property
    def resource(self) -> Traversable:
        """
        Return the importlib resource for this structure data file.

        :return: Traversable resource pointing to the packaged TSV file.
        """
        return files("natu.data.structures").joinpath(self.value)

    def open(self, encoding: str = "utf-8") -> IO:
        """
        Open this structure data file.

        :param encoding: Text encoding.
        :return: Open file handle.
        """
        return self.resource.open("r", encoding=encoding)

    def read_text(self, encoding: str = "utf-8") -> str:
        """
        Read this structure data file as text.

        :param encoding: Text encoding.
        :return: File contents as string.
        """
        return self.resource.read_text(encoding=encoding)

    @classmethod
    def from_name(cls, name: str) -> StructureData:
        """
        Return the packaged structure data file.
        :return: structure data.
        """

        return cls[name.upper()]


class AlignmentConfiguration(Enum):
    """
    Packaged alignment configuration files.

    Each enum member points to a YAML configuration file inside natu.data.alignment_configurations.
    """

    MATCH_MISMATCH = "match_mismatch.yaml"
    PARAS_BASED = "paras_based.yaml"
    ECFP = "ecfp.yaml"
    FEATMORGAN = "featmorgan.yaml"

    @property
    def resource(self) -> Traversable:
        """
        Return the importlib resource for this configuration file.

        :return: Traversable resource pointing to the packaged YAML file.
        """
        return files("natu.data.alignment_configurations").joinpath(self.value)

    def open(self, encoding: str = "utf-8") -> IO:
        """
        Open this configuration file.

        :param encoding: Text encoding.
        :return: Open file handle.
        """
        return self.resource.open("r", encoding=encoding)

    def read_text(self, encoding: str = "utf-8") -> str:
        """
        Read this configuration file as text.

        :param encoding: Text encoding.
        :return: File contents as string.
        """
        return self.resource.read_text(encoding=encoding)


class SubstitutionMatrix(Enum):
    """
    Packaged substitution matrix files.

    Each enum member points to a tab-separated substitution matrix file inside
    ``natu.data.substitution_matrices``.
    """

    MATCH_MISMATCH = "match_mismatch.txt"
    PARAS_BASED = "paras_based.txt"
    ECFP = "ecfp.txt"
    FEATMORGAN = "featmorgan.txt"

    @property
    def resource(self) -> Traversable:
        """
        Return the importlib resource for this substitution matrix file.

        :return: Traversable resource pointing to the packaged matrix file.
        """
        return files("natu.data.substitution_matrices").joinpath(self.value)

    def open(self, encoding: str = "utf-8") -> IO:
        """
        Open this substitution matrix file.

        :param encoding: Text encoding.
        :return: Open file handle.
        """
        return self.resource.open("r", encoding=encoding)

    def read_text(self, encoding: str = "utf-8") -> str:
        """
        Read this substitution matrix file as text.

        :param encoding: Text encoding.
        :return: File contents as string.
        """
        return self.resource.read_text(encoding=encoding)

    def get_config(self) -> AlignmentConfiguration:
        """
        Return the packaged alignment configuration file.
        :return: alignment configuration.
        """

        return AlignmentConfiguration[self.name]
