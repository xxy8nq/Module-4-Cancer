import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Prepare your data
# Assuming 'df' is your lung cancer dataset
df = pd.read_csv('data/umap_lung_hallmarks.csv')
# Features: Immune evasion and sustained growth gene expression
X = df[['immune_evasion_expression', 'sustained_growth_expression']] 
# Target: Benign (0) or Malignant (1)
y = df['tumor_status'] 

# 2. Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Choose and Fit a Model (using K-Nearest Neighbors as an example)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train) # Fit is used for in-sample data [cite: 20]

# 4. Predict the target variable
y_train_pred = model.predict(X_train) # For in-sample error [cite: 20]
y_val_pred = model.predict(X_val)    # For out-of-sample error 

# 5. Evaluate Performance
# Calculate In-sample vs Out-of-sample error 
in_sample_accuracy = accuracy_score(y_train, y_train_pred)
out_of_sample_accuracy = accuracy_score(y_val, y_val_pred)

print(f"In-sample Accuracy: {in_sample_accuracy:.2f}")
print(f"Out-of-sample Accuracy: {out_of_sample_accuracy:.2f}")

# 6. Visualize with a Confusion Matrix 
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Tumor Classification")
plt.show()