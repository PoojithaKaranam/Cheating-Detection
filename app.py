from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --------------------
# Load dataset & model
# --------------------
data = pd.read_csv("dataset/online_exam_cheating_dataset_balanced.csv")

# Encode categorical column if needed (answer_speed_variance)
if data["answer_speed_variance"].dtype == object:
    mapping = {"low": 0, "medium": 1, "high": 2}
    data["answer_speed_variance"] = data["answer_speed_variance"].map(mapping)

# Features and target
X = data.drop("cheating", axis=1)   # Note: dataset column is lowercase "cheating"
y = data["cheating"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        try:
            # Get form values
            tab_switches = float(request.form.get("tab_switches", 0))
            time_per_question = float(request.form.get("time_per_question", 0))
            idle_time = float(request.form.get("idle_time", 0))

            # Encode dropdowns
            answer_speed_variance_str = request.form.get("answer_speed_variance", "low")
            mapping = {"low": 0, "medium": 1, "high": 2}
            answer_speed_variance = mapping.get(answer_speed_variance_str, 0)

            copy_paste = int(request.form.get("copy_paste", 0))

            # Build features in same order as dataset
            features = [[
                tab_switches,
                time_per_question,
                idle_time,
                answer_speed_variance,
                copy_paste
            ]]

            # Predict
            prediction = model.predict(features)[0]

            if prediction == 1:
                prediction_text = "⚠️ Student is CHEATING"
            else:
                prediction_text = "✅ Student is NOT cheating"

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template("index.html", prediction=prediction_text)

# --------------------
# Run app
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
