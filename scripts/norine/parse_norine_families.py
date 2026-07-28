import json
from argparse import ArgumentParser, Namespace
from typing import Any
import os


def parse_argument() -> Namespace:
    parser = ArgumentParser(description="Parse Norine Families")
    parser.add_argument('-i', required=True, type=str,
                        help="Input file, output from 'fetch_norine_families.py'")
    parser.add_argument('-o', required=True, type=str,
                        help="Path to output directory")
    return parser.parse_args()



def write_families(families: dict[str, Any], out_dir: str) -> None:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for family, entries in families.items():
        out_path = os.path.join(out_dir, family)
        with open(out_path, "w") as out:
            for entry in entries:
                name = entry["norine"]["peptide"][0]["general"]["name"]
                composition = entry["norine"]["peptide"][0]["structure"]["composition"]
                print(type(composition))
                out.write(f"{name}\t{composition}\n")


def group_by_family(jsonl_path, composition_only=True):
    families = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["norine"]["peptide"]:
                fam = entry["norine"]["peptide"][0]["general"].get("family")  # confirm exact key name from a sample response first
                if fam:
                    has_comp = False

                    if "structure" in entry["norine"]["peptide"][0] and "composition" in entry["norine"]["peptide"][0]["structure"]:
                        has_comp = True
                    if composition_only and not has_comp:
                        continue

                    families.setdefault(fam, []).append(entry)
    return families


def main():
    args = parse_argument()

    families = group_by_family(args.i)
    write_families(families, args.o)

if __name__ == "__main__":
    main()