import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_csv("customer_data.csv")
categorical_cols = df.select_dtypes(include="object").columns
le_dict = {}
for col in categorical_cols:
    if col != "Churn": 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f" Model trained with accuracy: {accuracy:.2f}")
joblib.dump(model, "churn_model.pkl")
joblib.dump(le_dict, "encoders.pkl")
print(" Model and encoders saved successfully!")
print("Accuracy:", model.score(X_test, y_test))
joblib.dump(model, "churn_model.pkl")
joblib.dump(le_dict, "encoders.pkl")
joblib.dump(list(X.columns), "feature_order.pkl")
print(" Model, encoders, and feature order saved successfully!")
