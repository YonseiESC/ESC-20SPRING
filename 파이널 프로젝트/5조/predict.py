import argparse
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
from xgboost import XGBClassifier, Booster


warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Input your file name.")
parser.add_argument('file_name')

args = parser.parse_args()
file_name = args.file_name

df = pd.read_csv(file_name)

def y_check(df):
    return "class" in df.columns

if y_check(df):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
else:
    X = df

scaler = joblib.load('scaler.pkl')
X = scaler.transform(X)

#lr_clf = joblib.load('logistic.pkl')
#svm_clf = joblib.load('SVM_linear.pkl')
#rf_clf = joblib.load('forest_01.pkl')
xgb_clf = XGBClassifier(learning_rate=0.3, max_depth=10, min_child_weight=0.1, gamma=0.2, colsample_bytree=0.7)
booster = Booster()
booster.load_model('xgb.xgb')
xgb_clf._Booster = booster

#scores = pickle.load(open('scores.pkl', 'rb'))

#lr_probs = lr_clf.predict_proba(X)
#svm_probs = svm_clf.predict_proba(X)
#rf_probs = rf_clf.predict_proba(X)
xgb_probs = xgb_clf.predict_proba(X)

def predict():
    y_pred = np.empty(shape=(X.shape[0], ), dtype=np.int)
    for i in range(len(y_pred)):
        pos = xgb_probs[i, 1]
        neg = xgb_probs[i, 0]
        #pos = lr_probs[i, 1] * scores[0] + svm_probs[i, 1] * scores[1] + rf_probs[i, 1] * scores[2] + xgb_probs[i, 1] * scores[3]
        #neg = lr_probs[i, 0] * scores[0] + svm_probs[i, 0] * scores[1] + rf_probs[i, 0] * scores[2] + xgb_probs[i, 0] * scores[3]
        if pos > neg:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(file_name.rstrip(".csv") + "_" + "prediction.csv", index=False)

    return y_pred

if __name__ == "__main__":
    y_pred = predict()















