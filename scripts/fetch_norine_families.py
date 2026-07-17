import requests
import time
import json
from sys import argv


BASE = "http://norine.univ-lille.fr/norine/rest/id/json/{}"

def fetch_peptide(norine_id: int, retries=3, pause=0.3):
    """norine_id as int, e.g. 123 for NOR00123 -- the API accepts unpadded IDs."""
    url = BASE.format(norine_id)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            return None  # likely a gap in ID numbering (deleted/never used)
        except requests.RequestException:
            time.sleep(pause * (attempt + 1))
    return None

def fetch_all_peptides(max_id=2020, out_path="norine_peptides.jsonl"):
    with open(out_path, "w") as f:
        for i in range(1, max_id + 1):
            data = fetch_peptide(i)
            if data:
                f.write(json.dumps(data) + "\n")
            time.sleep(0.2)  # be polite -- avoid hammering their server
    return out_path

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

def get_max_id(data):
    ids = []
    for entry in data:
        id_str = entry.get("general", {}).get("id", "")
        # IDs look like "NOR00001" -- strip the prefix and leading zeros
        digits = id_str.replace("NOR", "")
        if digits.isdigit():
            ids.append(int(digits))
    return max(ids) if ids else None



if __name__ == "__main__":
    families = group_by_family(argv[1])
    fams = sorted(families.keys())
    for fam in fams:
        data = families[fam]
        print(fam)
        for entry in data:
            print(entry["norine"]["peptide"][0]["general"]["name"])
            print(entry["norine"]["peptide"][0]["structure"]["composition"])
        print('\n')



    # for fam in fams:
    #     print(fam)
    # fetch_all_peptides(out_path=argv[1])
    # r = requests.get("https://norine.univ-lille.fr/norine/rest/peptides/json/smiles", timeout=60)
    # data = r.json()["peptides"]
    # max_id = get_max_id(data)
    # print(f"Max Norine ID: NOR{max_id:05d}")
    # print(f"Total entries returned: {len(data)}")

