import joblib
import sklearn
model = joblib.load("./models/decision_tree1.pkl")
print(model)