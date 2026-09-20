import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load the customer churn dataset
df = pd.read_csv("WA_FnUseC_TelcoCustomerChurn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove customer ID because it is not a predictive feature
df = df.drop(columns=["customerID"])

# Convert target values to numbers
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Separate input features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nData preparation completed!")
print("Cleaned dataset shape:", df.shape)
print("Input features:", X.shape[1])
print("Target distribution:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nTrain-test split completed!")
print("Training customers:", X_train.shape[0])
print("Testing customers:", X_test.shape[0])

print("\nNumerical features:", numerical_features)
print("\nCategorical features:", categorical_features)

# Prepare numerical features
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Prepare categorical features
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")

# Combine preprocessing and Logistic Regression
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train the model using training data
print("\nTraining the customer churn model...")

model_pipeline.fit(X_train, y_train)

print("Model training completed successfully!")

# Make predictions on testing data
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and preprocessing pipeline
joblib.dump(model_pipeline, "churn_model.joblib")

print("\nTrained model saved successfully!")
print("Model file: churn_model.joblib")