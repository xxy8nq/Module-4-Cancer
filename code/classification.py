import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Setup paths
repo_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
data_dir = repo_root / "data"

# Load training-only data
hallmarks = pd.read_csv(data_dir / "umap_lung_hallmarks.csv", index_col=0)
metadata = pd.read_csv(data_dir / "TRAINING_SET_GSE62944_metadata.csv", index_col=0)

# Merge hallmark means with clinical data on the sample ID
df = hallmarks.join(metadata[["tumor_status"]], how="inner")

# Keep only the two tumor-status labels used for the classification task
df = df[df["tumor_status"].isin(["TUMOR FREE", "WITH TUMOR"])].copy()
df["target"] = df["tumor_status"].map({"TUMOR FREE": 0, "WITH TUMOR": 1})

# Define features and target
X = df[["immune_hallmark_mean", "sustained_hallmark_mean"]]
y = df["target"]

# Split the training set into training and validation partitions only
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# Train a simple tree to keep the model interpretable
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_preds)
val_accuracy = accuracy_score(y_val, val_preds)
train_balanced_accuracy = balanced_accuracy_score(y_train, train_preds)
val_balanced_accuracy = balanced_accuracy_score(y_val, val_preds)
generalization_gap = train_balanced_accuracy - val_balanced_accuracy

print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Validation accuracy: {val_accuracy:.2f}")
print(f"Training balanced accuracy: {train_balanced_accuracy:.2f}")
print(f"Validation balanced accuracy: {val_balanced_accuracy:.2f}")
print(f"Generalization gap (balanced accuracy): {generalization_gap:.2f}")
print("\nValidation classification report:")
print(classification_report(y_val, val_preds, target_names=["TUMOR FREE", "WITH TUMOR"]))

# Validation confusion matrix shows the main error types
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