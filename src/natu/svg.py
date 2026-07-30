"""Module for visualizing MSA as SVG."""

import re
from html import escape

from natu.constants import GAP_REPR


PROTEINOGENIC_AMINO_ACID_NAMES = {
    "alanine": "ALA",
    "arginine": "ARG",
    "asparagine": "ASN",
    "aspartic acid": "ASP",
    "aspartate": "ASP",
    "cysteine": "CYS",
    "glutamic acid": "GLU",
    "glutamate": "GLU",
    "glutamine": "GLN",
    "glycine": "GLY",
    "histidine": "HIS",
    "isoleucine": "ILE",
    "leucine": "LEU",
    "lysine": "LYS",
    "methionine": "MET",
    "phenylalanine": "PHE",
    "proline": "PRO",
    "serine": "SER",
    "threonine": "THR",
    "tryptophan": "TRP",
    "tyrosine": "TYR",
    "valine": "VAL",
}


def summarize_block_name(block: str | None) -> str:
    """
    Convert a full monomer name to a fixed 3-character uppercase label.

    Proteinogenic amino acid full names are protected and map to standard
    3-letter amino acid codes. All other names use their first 3 alphanumeric
    characters, uppercased.
    """
    if block is None:
        return "---"

    name = str(block).strip()
    if not name:
        return "---"

    protected_name = PROTEINOGENIC_AMINO_ACID_NAMES.get(name.lower())
    if protected_name is not None:
        return protected_name

    alnum = re.sub(r"[^A-Za-z0-9]", "", name).upper()

    if not alnum:
        return "---"

    return alnum[:3].ljust(3)


def msa_to_svg(msa: list[tuple[str, list[str]]]) -> str:
    """
    Convert MSA to SVG.

    :param msa: List of (header, sequence) tuples.
    :return: SVG string.
    """
    if not msa:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1" '
            'viewBox="0 0 1 1"></svg>'
        )

    block_width = 42
    block_height = 28
    row_gap = 8
    padding = 16

    # Monospace char width approximation at 12px font-size (~0.6em advance width)
    font_size = 12
    char_width = font_size * 0.6
    label_gap = 16  # breathing room between longest label and first block

    max_header_len = max(len(escape(header)) for header, _ in msa)
    label_width = max_header_len * char_width + label_gap

    max_cols = max(len(sequence) for _, sequence in msa)
    row_height = block_height + row_gap

    width = padding * 2 + label_width + max_cols * block_width
    height = padding * 2 + len(msa) * row_height - row_gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text { font-family: Menlo, Consolas, monospace; font-size: 12px; }",
        ".seq-label { fill: #222; dominant-baseline: middle; }",
        ".block-text { fill: #111; text-anchor: middle; dominant-baseline: middle; }",
        ".block { stroke: #000000; stroke-width: 1; rx: 4; ry: 4; }",
        "</style>",
    ]

    for row_index, (header, sequence) in enumerate(msa):
        y = padding + row_index * row_height
        label_y = y + block_height / 2

        parts.append(
            f'<text class="seq-label" x="{padding}" y="{label_y}">'
            f"{escape(header)}"
            "</text>"
        )

        for col_index in range(max_cols):
            block = sequence[col_index] if col_index < len(sequence) else None

            if block == GAP_REPR:
                continue

            label = summarize_block_name(block)
            fill = "#ffffff"

            x = padding + label_width + col_index * block_width
            text_x = x + block_width / 2
            text_y = y + block_height / 2

            parts.append(
                f'<rect class="block" x="{x}" y="{y}" '
                f'width="{block_width}" height="{block_height}" fill="{fill}" />'
            )
            parts.append(
                f'<text class="block-text" x="{text_x}" y="{text_y}">'
                f"{escape(label)}"
                "</text>"
            )

    parts.append("</svg>")

    return "\n".join(parts)
