import joblib

loaded_model = joblib.load("best_model.pkl")
new_query = "/**/and/**/1/**/ = /**/1"
prediction = loaded_model.predict([new_query])
print(prediction)
