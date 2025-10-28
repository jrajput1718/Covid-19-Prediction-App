import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("Covid Dataset.csv")
print(" Dataset loaded successfully!\n", df.head())

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Inspect available columns
print("\n Available Columns:", list(df.columns))

# Map alternate names (handles variations)
possible_names = {
    'Fever': ['Fever', 'fever'],
    'Dry Cough': ['Dry Cough', 'Cough', 'dry cough'],
    'Fatigue': ['Fatigue', 'Fatigue ', 'Tiredness', 'Weakness'],
    'Breathing Problem': ['Breathing Problem', 'Breathing Difficulty', 'Shortness of Breath'],
    'COVID-19': ['COVID-19', 'COVID', 'Corona Result']
}


# Find actual matching column names
selected_features = {}
for key, options in possible_names.items():
    for opt in options:
        if opt in df.columns:
            selected_features[key] = opt
            break

print("\n Matched Columns:", selected_features)

# Check if required columns exist
required = ['Fever', 'Dry Cough', 'Fatigue', 'Breathing Problem', 'COVID-19']
missing = [col for col in required if col not in selected_features]
if missing:
    raise KeyError(f" Missing required columns in dataset: {missing}")

# Select columns
df = df[[selected_features[c] for c in required]]

# Rename to standard names
df.columns = required

# Split data
X = df.drop('COVID-19', axis=1)
y = df['COVID-19']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel='linear', probability=True)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Save best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
joblib.dump(best_model, "covid_model.pkl")
print(f" Best model saved: {best_model_name}")

# Plot comparison
plt.figure(figsize=(6, 4))
plt.bar(accuracies.keys(), accuracies.values(), color=['green', 'blue', 'orange'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("model_accuracy.png")
plt.show()
