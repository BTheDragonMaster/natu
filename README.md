# NATU: NRPS Alignment of Thiotemplated Units

## Installation

Create a conda environment and install the dependencies:

```bash
conda create -n natu python=3.10
conda activate natu
pip install .
```

## Usage

```bash
natu -s match_mismatch -f data/test.fa -o out.msa
```