import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

def train_models():
    df = pd.read_csv("data.csv")

    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    X = df[["age", "gender", "time_spent"]]
    y = df["click"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    acc = {
        "Logistic Regression": accuracy_score(y_test, lr.predict(X_test)),
        "Random Forest": accuracy_score(y_test, rf.predict(X_test)),
        "XGBoost": accuracy_score(y_test, xgb_model.predict(X_test))
    }

    return lr, rf, xgb_model, le, acc