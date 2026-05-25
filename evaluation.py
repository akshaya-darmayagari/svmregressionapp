import pandas as pd
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["Progression"] = diabetes.target

# Split features and target
X = df.drop("Progression", axis=1)
y = df["Progression"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = SVR(kernel="rbf", C=100.0, epsilon=0.1)
model.fit(X_train, y_train)

# Save artifacts
pickle.dump(model, open("model_svr.pkl", "wb"))
pickle.dump(scaler, open("scaler_svr.pkl", "wb"))

print("SVR Model Saved Successfully")