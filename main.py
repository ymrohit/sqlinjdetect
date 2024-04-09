import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the dataset from a CSV file
df = pd.read_csv("Dataset.csv")
df1 = pd.read_csv("sqli.csv", encoding='utf-16')
df2 = pd.read_csv("sqliv2.csv", encoding='utf-16')
# Split the dataset into features (X) and target (y)
# Rename columns for consistency
df.columns = ['Query', 'Label']
df1.columns = ['Query', 'Label']
df2.columns = ['Query', 'Label']

# Now concatenate all three dataframes
df_final = pd.concat([df, df1, df2], ignore_index=True)
df_final['Label'].fillna(0, inplace=True)
df_final.dropna(subset=['Query','Label'], inplace=True)
print(df_final.shape)

X = df_final["Query"]
y = df_final["Label"]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and their hyperparameter variations
# Define the models and their hyperparameter variations
models = [
    ("Naive Bayes", MultinomialNB(), {
        "classifier__alpha": [0.1, 0.5, 1.0],
        "classifier__fit_prior": [True, False]
    }),
    ("Logistic Regression", LogisticRegression(), {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["liblinear", "saga"]
    }),
    ("SVM", SVC(), {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["linear", "rbf"]
    }),
    #("Random Forest", RandomForestClassifier(), {"classifier__n_estimators": [50, 100, 200],"classifier__max_depth": [None, 5, 10]})
]

# Iterate over the models
best_model = None
best_accuracy = 0
best_params = None

for model_name, model, param_grid in models:
    # Create a pipeline with CountVectorizer and the model
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", model)
    ])

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model_in_grid = grid_search.best_estimator_
    best_params_in_grid = grid_search.best_params_

    # Predict on the testing set using the best model
    y_pred = best_model_in_grid.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Best Accuracy: {accuracy}")
    print(f"{model_name} Best Parameters: {best_params_in_grid}")

    # Update the overall best model if the current model has higher accuracy
    if accuracy > best_accuracy:
        best_model = best_model_in_grid
        best_accuracy = accuracy
        best_params = best_params_in_grid

print(f"\nBest Model: {best_model.named_steps['classifier']}")
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Parameters: {best_params}")

# Test the best model with a new query
new_query = "SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1'"
prediction = best_model.predict([new_query])
print("Prediction:", prediction)
import joblib
joblib.dump(best_model, "best_model.pkl")

'''
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# Convert the trained model to ONNX format
initial_type = [('input', StringTensorType([None, 1]))]
onx = convert_sklearn(best_model, initial_types=initial_type)

# Save the ONNX model to a file
with open("best_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
'''
