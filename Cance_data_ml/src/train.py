import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from preprocessing import load_data, remove_outliers_zscore, cap_outliers_iqr

df = load_data("data/raw/Cancer_Data.csv")

X = df.drop(columns="diagnosis")
y = df["diagnosis"]

X, y = remove_outliers_zscore(X, y)
X = cap_outliers_iqr(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=7, weights="distance")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
