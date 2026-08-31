import sys

from pikachu.general import read_smiles
from pikachu.drawing.drawing import Drawer, draw_multiple, Options
from pikachu.drawing.colours import get_hex
from pikachu.chem.structure import Structure

from get_substrate_variants import parse_smiles

def svg_from_structure(structure: Structure, svg_out: str, colour: str | None = None, kekulise: bool = True):
    """
    Save structure drawing of SMILES string to .svg

    Input:
    smiles: str, SMILES string
    svg_out: str, output file name, should end in .svg

    """
    options = Options()

    drawer = Drawer(structure, options=options, coords_only=True, kekulise=kekulise)

    if colour is not None:
        if not colour.startswith('#'):
            colour = get_hex(colour)
        for atom in drawer.structure.graph:
            atom.draw.colour = colour

    drawer.write_svg(svg_out)

for name, structure in parse_smiles(sys.argv[1]).items():
    svg_from_structure(structure, f"{name}.svg", colour=sys.argv[2])