import os
import requests
import time
import json
from sys import argv


BASE = "https://norine.univ-lille.fr/norine/rest/id/json/{}"
SMILES_URL = "https://norine.univ-lille.fr/norine/rest/peptides/json/smiles"

# Norine refuses plain HTTP and anonymous clients (the default
# python-requests User-Agent) with 403, so identify the script.
# Set NORINE_CONTACT to your e-mail address before running.
HEADERS = {
    "User-Agent": "natu-research-script/0.1 "
                  f"(contact: {os.environ.get('NORINE_CONTACT', 'unset')})",
    "Accept": "application/json",
}

def fetch_peptide(norine_id: int, retries=3, pause=0.3):
    """norine_id as int, e.g. 123 for NOR00123 -- the API accepts unpadded IDs."""
    url = BASE.format(norine_id)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 403:
                raise RuntimeError(f"Norine returned 403 Forbidden for {url}")
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

    out_path = argv[1] if len(argv) > 1 else "norine_peptides.jsonl"

    r = requests.get(SMILES_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()["peptides"]
    max_id = get_max_id(data)
    print(f"Max Norine ID: NOR{max_id:05d}")
    print(f"Total entries returned: {len(data)}")

    fetch_all_peptides(max_id=max_id, out_path=out_path)
    with open(out_path) as f:
        print(f"Wrote {sum(1 for _ in f)} peptides to {out_path}")

