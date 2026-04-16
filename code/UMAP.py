from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap


def parse_hallmark_genes(core_file: Path, hallmark_name: str) -> list[str]:
	"""Parse one hallmark row from the tab-delimited core hallmark file."""
	with core_file.open("r", encoding="utf-8") as handle:
		for line in handle:
			parts = [piece.strip() for piece in line.strip().split("\t") if piece.strip()]
			if parts and parts[0] == hallmark_name:
				return list(dict.fromkeys(parts[1:]))
	return []


def resolve_immune_genes(expression_index: pd.Index, immune_raw_genes: list[str]) -> list[str]:
	"""Map aliases/split genes from the immune list to genes present in expression data."""
	aliases = {
		"HER2": ["HER2", "ERBB2"],
		"RAS": ["KRAS", "HRAS", "NRAS", "RAS"],
	}

	resolved = []
	for gene in immune_raw_genes:
		if "/" in gene:
			candidates = gene.split("/")
		else:
			candidates = aliases.get(gene, [gene])

		for candidate in candidates:
			if candidate in expression_index:
				resolved.append(candidate)

	return list(dict.fromkeys(resolved))


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Data sources
expression_df = pd.read_csv(DATA_DIR / "TRAINING_SET_GSE62944_subsample_log2TPM.csv", index_col=0)
metadata_df = pd.read_csv(DATA_DIR / "TRAINING_SET_GSE62944_metadata.csv", index_col=0)

# Cancer type of interest for this project (LUAD only)
target_cancer_type = "LUAD"

if target_cancer_type not in metadata_df["cancer_type"].unique():
	raise ValueError("No LUAD samples found in metadata.")

subset_metadata = metadata_df[metadata_df["cancer_type"] == target_cancer_type].copy()

# Hallmark 1: immune evasion genes from local immune gene file
immune_namespace = runpy.run_path(str(DATA_DIR / "immuneevasiongenes.csv"))
immune_raw_genes = [row[0] for row in immune_namespace["gene_data"][1:]]
immune_genes = resolve_immune_genes(expression_df.index, immune_raw_genes)

# Hallmark 2: sustained proliferative signaling genes from core hallmark file
core_file = BASE_DIR / "Menyhart_JPA_CancerHallmarks_core.txt"
sustained_all = parse_hallmark_genes(core_file, "SUSTAINING PROLIFERATIVE SIGNALING")
sustained_available = [gene for gene in sustained_all if gene in expression_df.index]

# Keep at least 10 genes from each hallmark; use many sustained genes to exceed 100 total features.
MAX_SUSTAINED_GENES = 140
sustained_genes = sustained_available[:MAX_SUSTAINED_GENES]

if len(immune_genes) < 10:
	raise ValueError(f"Need >=10 immune-evasion genes, found {len(immune_genes)}.")
if len(sustained_genes) < 10:
	raise ValueError(f"Need >=10 sustained-signaling genes, found {len(sustained_genes)}.")

genes_of_interest = list(dict.fromkeys(immune_genes + sustained_genes))

# Reduce expression matrix to selected samples (columns) and selected genes (rows)
reduced_expression = expression_df.loc[genes_of_interest, subset_metadata.index]

# UMAP input should be samples x genes
X = reduced_expression.T.copy()
X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(X_scaled)

umap_df = pd.DataFrame(embedding, index=X.index, columns=["UMAP1", "UMAP2"])
umap_df = umap_df.join(subset_metadata[["cancer_type"]])

# Project-relevant coloring features
umap_df["immune_hallmark_mean"] = X[immune_genes].mean(axis=1)
umap_df["sustained_hallmark_mean"] = X[sustained_genes].mean(axis=1)
key_gene = "EGFR" if "EGFR" in X.columns else immune_genes[0]
umap_df["key_gene_expression"] = X[key_gene]

print("Samples retained:", X.shape[0])
print("Cancer type included:", target_cancer_type)
print("Immune-evasion genes used:", len(immune_genes))
print("Sustained-signaling genes used:", len(sustained_genes))
print("Total unique genes used:", len(genes_of_interest))
print("Key gene for coloring:", key_gene)

# Save embedding for downstream notebook/modeling use
out_file = DATA_DIR / "umap_lung_hallmarks.csv"
umap_df.to_csv(out_file)
print("Saved:", out_file)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(
	data=umap_df,
	x="UMAP1",
	y="UMAP2",
	hue="cancer_type",
	palette="Set2",
	alpha=0.9,
	s=60,
	ax=axes[0],
)
axes[0].set_title("UMAP by Cancer Type")

sc1 = axes[1].scatter(
	umap_df["UMAP1"],
	umap_df["UMAP2"],
	c=umap_df["immune_hallmark_mean"],
	cmap="viridis",
	s=60,
	alpha=0.9,
)
axes[1].set_title("UMAP by Immune Hallmark Mean")
fig.colorbar(sc1, ax=axes[1], label="Mean expression")

sc2 = axes[2].scatter(
	umap_df["UMAP1"],
	umap_df["UMAP2"],
	c=umap_df["key_gene_expression"],
	cmap="magma",
	s=60,
	alpha=0.9,
)
axes[2].set_title(f"UMAP by {key_gene} Expression")
fig.colorbar(sc2, ax=axes[2], label="log2 TPM")

for ax in axes:
	ax.set_xlabel("UMAP1")
	ax.set_ylabel("UMAP2")

plt.tight_layout()
plt.show()