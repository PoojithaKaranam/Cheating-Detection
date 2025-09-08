from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("online_exam_cheating_dataset_balanced.csv")

# Prepare features and target
X = data.drop("Cheating", axis=1)
y = data["Cheating"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form data
            features = [float(request.form.get(col, 0)) for col in X.columns]
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, columns=X.columns)

if __name__ == "__main__":
    app.run(debug=True)
