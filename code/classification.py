import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
repo_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
data_dir = repo_root / "data"

# Load training-only data
hallmarks = pd.read_csv(data_dir / "umap_lung_hallmarks.csv", index_col=0)
metadata = pd.read_csv(data_dir / "TRAINING_SET_GSE62944_metadata.csv", index_col=0)

# Merge hallmark means with clinical data on the sample ID
df = pd.merge(hallmarks, metadata[['sample', 'tumor_status']], left_on='sample', right_on='sample')

# Keep only the two tumor-status labels used for the classification task
df = df[df["tumor_status"].isin(["TUMOR FREE", "WITH TUMOR"])].copy()
df["target"] = df["tumor_status"].map({"TUMOR FREE": 0, "WITH TUMOR": 1})

# Define features and target
X = df[["immune_hallmark_mean", "sustained_hallmark_mean"]]
y = df["target"]

# 70/30 Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the Decision Tree
# We limit max_depth to avoid the "overfitting" mentioned in Lecture 5
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Calculate Accuracy
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

print(f"In-sample Accuracy: {accuracy_score(y_train, train_preds):.2f}")
print(f"Out-of-sample Accuracy: {accuracy_score(y_val, val_preds):.2f}")

import matplotlib.pyplot as plt

# Generate Confusion Matrix
cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TUMOR FREE", "WITH TUMOR"])

disp.plot(cmap="Blues")
plt.title("Validation Confusion Matrix: LUAD Tumor Status Prediction")
plt.tight_layout()
plt.show()

# Tree visualization for interpretation
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=["Immune Hallmark Mean", "Growth Hallmark Mean"],
    class_names=["TUMOR FREE", "WITH TUMOR"],
    filled=True,
    rounded=True,
    fontsize=12,
)

plt.title("Decision Tree Logic for LUAD Tumor Status")
plt.tight_layout()
plt.show()