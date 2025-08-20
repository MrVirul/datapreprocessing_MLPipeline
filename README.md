# 🎯 Student Result Prediction using Random Forest Classifier

This project applies a **Random Forest Classifier** to predict whether a student will **Pass** or **Fail** based on their features (e.g., Age, Quiz Score, etc.).  

The pipeline includes **feature preparation, train-test split, model training, evaluation, and feature importance analysis**.

---

## 📂 Steps in the Pipeline

### 1️⃣ Prepare Features (X) and Label (y)
We remove irrelevant or target columns:
- `StudentID` → Not useful for prediction  
- `Country` → Categorical (not encoded for this task)  
- `Result` → Target variable (label)  

```python
X = df.drop(columns=["StudentID", "Country", "Result"])
y = df["Result"]
2️⃣ Train-Test Split
Split the dataset into 80% training and 20% testing.
Random state ensures reproducibility.

python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
3️⃣ Build & Train Random Forest Classifier
python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
4️⃣ Model Evaluation
Use accuracy score and classification report to check performance.

python
Copy
Edit
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
5️⃣ Feature Importance
Visualize which features are most important in predicting the result.

python
Copy
Edit
import matplotlib.pyplot as plt
import seaborn as sns

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")

plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
📊 Example Outputs
Accuracy: e.g., 0.87 (87%)

Classification Report: Precision, Recall, and F1-score for Pass and Fail

Feature Importance Plot: Shows which features most influenced the model

🚀 Next Steps
Encode categorical features like Country

Hyperparameter tuning for Random Forest (GridSearchCV / RandomizedSearchCV)

Compare with other models (Logistic Regression, XGBoost, etc.)

📌 Author
👤 Virul Methdinu Meemana
📍 Sri Lanka | Information & Technology Student at SLIIT

🔗 GitHub | LinkedIn
