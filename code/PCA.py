import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. SETUP PATHS
repo_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = repo_root / "data"

# 2. LOAD DATA
data = pd.read_csv(DATA_DIR / "TRAINING_SET_GSE62944_subsample_log2TPM.csv", index_col=0)
metadata = pd.read_csv(DATA_DIR / "TRAINING_SET_GSE62944_metadata.csv", index_col=0)

# 3. DEFINE GENE LISTS (From your specific CSV/List)
growth_genes = [
    "EGFR", "KRAS", "NRAS", "HRAS", "BRAF", "MYC", "ERBB2", "ERBB3", "MET", 
    "PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "MTOR", "MAPK1", "MAPK3", 
    "CCND1", "CDK4", "CDK2", "SOS1", "GRB2", "RAF1", "FGFR1", "PDGFA"
]

immune_genes = [
    "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "IDO1", "STAT1", "STAT3",
    "HLA-A", "HLA-B", "B2M", "JAK1", "JAK2", "IFNGR1", "IL6", "CD80", "CD86",
    "PTPN6", "LCK", "ZAP70", "CBL", "CBLB", "TRAF6", "PRKCB", "VAV1"
]

# Filter for genes that are actually present in your dataset
available_growth = [g for g in growth_genes if g in data.index]
available_immune = [g for g in immune_genes if g in data.index]
project_genes = list(set(available_growth + available_immune))

# 4. SUBSET FOR LUAD
luad_samples = metadata[metadata['cancer_type'] == 'LUAD'].index
df = data.loc[project_genes, luad_samples].T 

# 5. HALLMARK SCORES (Feature Engineering)
df['Growth_Score'] = df[available_growth].mean(axis=1)
df['Immune_Score'] = df[available_immune].mean(axis=1)

# 6. VISUALIZATION: REGRESSION (Growth vs Immune)
sns.regplot(data=df, x='Growth_Score', y='Immune_Score', scatter_kws={'alpha':0.3})
plt.title("LUAD Hallmark Correlation: Growth vs. Immune Evasion")
plt.show()

# 7. PCA (Unsupervised Learning)
# Scale the raw gene features (excluding the scores)
genes_only = df.drop(columns=['Growth_Score', 'Immune_Score'])
x_scaled = StandardScaler().fit_transform(genes_only)

pca = PCA(n_components=2)
coords = pca.fit_transform(x_scaled)
pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=df.index)

# 8. VISUALIZATION: PCA (Colored by Immune Score)
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df['Immune_Score'], palette='magma')
plt.title("PCA of LUAD: Variation in Growth and Immune Genes")
plt.show()

# Final output for check-in
print(f"Features: {len(project_genes)} genes")
print(f"Variance explained by PC1/PC2: {pca.explained_variance_ratio_}")