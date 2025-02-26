import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from joblib import dump
from utils.preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('../data/synthetic_transactions.csv')

# Train XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
model.save_model('../model/xgboost_model.json')
dump(scaler, '../model/scaler.joblib')
