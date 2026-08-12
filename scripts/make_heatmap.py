"""Make heatmap of substitution matrix"""

from argparse import ArgumentParser, Namespace

import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Print specific element from substitution matrix")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to substitution matrix")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to output figure")

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t", index_col=0)
    distance_matrix = df.max().max() - df
    condensed_dist = squareform(distance_matrix.values, checks=False)

    # 3. Compute the hierarchical clustering linkage (e.g., using 'average' or 'complete' method)
    linkage_matrix = linkage(condensed_dist, method='average')

    # 4. Generate the Clustered Heatmap
    g = sns.clustermap(
        df,  # Display your original log-odds scores
        row_linkage=linkage_matrix,  # Cluster rows using your distance matrix
        col_linkage=linkage_matrix,  # Cluster columns identically (symmetrical)
        cmap="vlag",  # Choose a clear color palette
        xticklabels=True,  # FORCE ALL COLUMN LABELS TO SHOW
        yticklabels=True,
        cbar_pos=(0.9, 0.3, 0.03, 0.6),
        dendrogram_ratio=0.06,
        figsize=(15, 15)  # Size of the entire visual block
    )

    font_size = round(750 / len(linkage_matrix))
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='right', fontsize=font_size)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=font_size)
    plt.savefig(args.output)

if __name__ == "__main__":
    main()