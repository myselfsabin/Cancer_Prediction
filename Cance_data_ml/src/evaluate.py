import joblib
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import load_data, remove_outliers_zscore, cap_outliers_iqr

df = load_data("data/raw/Cancer_Data.csv")

X = df.drop(columns="diagnosis")
y = df["diagnosis"]

X, y = remove_outliers_zscore(X, y)
X = cap_outliers_iqr(X)

_, x_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/knn_model.pkl")

x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="M"))
print("Recall:", recall_score(y_test, y_pred, pos_label="M"))
print("F1:", f1_score(y_test, y_pred, pos_label="M"))
