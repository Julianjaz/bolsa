from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

train_df = pd.read_parquet("train_data.parquet")
test_df = pd.read_parquet("test_data.parquet")

features = [
    "Open", "High", "Low", "Close", "Volume",
    "daily_return", "volatility_5d", "ma_5", "ma_10",
    "rsi_14", "volume_change",
    "sentiment_value", "sentiment_score", "sentiment_score_ma3",
    "day_of_week"
]

X_train = train_df[features]
y_train = train_df["target"]

X_test = test_df[features]
y_test = test_df["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)




y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
