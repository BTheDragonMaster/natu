"""Tests for natu.constants: packaged config/data-file access.

These exercise the real packaged resources (YAML configs, TSV structure
data, TXT substitution matrices) that ship inside natu/data/ -- if one of
these enum members points at a file that's been renamed or removed, these
tests catch it, since matrix.py/cli.py only fail much later and less
clearly (deep inside pandas/yaml parsing) when a resource goes missing.

Layout: one test class per function under test, named Test<FunctionName>.
None of what's covered here is a bare function though -- GAP_REPR is a
module constant and the rest are enum classes -- so each gets one test
class named after it instead (TestGapRepr, TestAlignmentConfiguration,
etc.), the same treatment as Converter in test_pairwise.py.
"""

from __future__ import annotations

import pytest
import yaml

from natu.constants import AlignmentConfiguration, GAP_REPR, StructureData, SubstitutionMatrix


class TestGapRepr:
    def test_is_a_stable_sentinel_string(self):
        assert GAP_REPR == "GAP"


class TestAlignmentConfiguration:
    @pytest.mark.parametrize("member", list(AlignmentConfiguration))
    def test_every_file_parses_as_yaml(self, member):
        content = member.read_text()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert parsed  # not empty


class TestSubstitutionMatrix:
    @pytest.mark.parametrize("member", list(SubstitutionMatrix))
    def test_every_file_is_non_empty(self, member):
        content = member.read_text()
        assert len(content) > 0
        # tab-separated matrix files: first line is a header row
        first_line = content.splitlines()[0]
        assert "\t" in first_line

    @pytest.mark.parametrize("member", list(SubstitutionMatrix))
    def test_maps_to_a_same_named_alignment_configuration(self, member):
        config = member.get_config()
        assert config.name == member.name


class TestStructureData:
    def test_from_name_is_case_insensitive(self):
        assert StructureData.from_name("default") is StructureData.DEFAULT
        assert StructureData.from_name("DEFAULT") is StructureData.DEFAULT
        assert StructureData.from_name("paras") is StructureData.PARAS

    def test_from_name_rejects_unknown_name(self):
        with pytest.raises(KeyError):
            StructureData.from_name("not_a_real_structure_set")

    def test_open_returns_a_readable_handle(self):
        with StructureData.DEFAULT.open() as handle:
            first_line = handle.readline()
        assert first_line  # got real content, not an empty/missing file
