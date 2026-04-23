# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
# %%
# Load the data
####################################################
data = pd.read_csv(
    DATA_DIR / "TRAINING_SET_GSE62944_subsample_log2TPM.csv",
    index_col=0,
    header=0,
)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    DATA_DIR / "TRAINING_SET_GSE62944_metadata.csv",
    index_col=0,
    header=0,
)
print(data.head())

# %%
# Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
####################################################
cancer_type = 'LUAD'  # Lung adenocarcinoma

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
LUAD_data = data[cancer_samples]

# %%
# Subset by index (genes)
####################################################w
# This file is a small text artifact, not a real CSV, so we keep the gene list explicit.
raw_gene_list = [
    "MYC",
    "RAS",
    "EGFR",
    "PIK3CA",
    "BRAF",
    "HER2",
    "CTNNB1",
    "STAT3",
    "PTEN",
    "TP53",
    "RB1",
    "APC",
    "BRCA1/2",
    "STK11",
    "SMAD4",
    "ATM",
]

from pathlib import Path
repo_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
hallmarks = pd.read_table(repo_root / "Menyhart_JPA_CancerHallmarks_core.txt", header=None, index_col=0)


# Extract gene names. If 'BRCA1/2' is in there, split it into two.
desired_gene_list = []
for gene in raw_gene_list:
    if '/' in gene:
        desired_gene_list.extend(gene.split('/'))
    else:
        desired_gene_list.append(gene)



# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
available_genes = [gene for gene in desired_gene_list if gene in LUAD_data.index]
LUAD_gene_data = LUAD_data.loc[available_genes]
print(LUAD_gene_data.head())

# %%
# Basic statistics on the subsetted data
####################################################
print(LUAD_gene_data.describe())
print(LUAD_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(LUAD_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(LUAD_gene_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())
# %%
# Merging datasets
####################################################
# Merge the subsetted expression data with metadata for LUAD samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
LUAD_metadata = metadata_df.loc[cancer_samples]
LUAD_merged = LUAD_gene_data.T.merge(LUAD_metadata, left_index=True, right_index=True)
print(LUAD_merged.head())

# %%
# Plotting
####################################################
# Boxplot of EGFR expression in LUAD samples using SEABORN
# Works really well with pandas dataframes, because most methods allow you to pass in a dataframe directly
sns.boxplot(data=LUAD_merged, x="gender", y='EGFR')
plt.title("EGFR Expression by Gender in LUAD Samples")
plt.show()

# Boxplot of MYC and EGFR expression in LUAD samples using PANDAS directly
gene_plot_columns = [gene for gene in ["MYC", "EGFR"] if gene in LUAD_merged.columns]
LUAD_merged[gene_plot_columns].plot.box()
plt.title("MYC and EGFR Expression in LUAD Samples")
plt.show()

# %%

