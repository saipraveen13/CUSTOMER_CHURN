from flask import Flask, render_template, request
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
 
        input_data = request.form.to_dict()


        input_data["SeniorCitizen"] = int(input_data["SeniorCitizen"])
        input_data["tenure"] = int(input_data["tenure"])
        input_data["MonthlyCharges"] = float(input_data["MonthlyCharges"])
        input_data["TotalCharges"] = float(input_data["TotalCharges"])

        df = pd.DataFrame([input_data])

        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        df = df[feature_order]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] 

        if prediction == 1:
            result = f"❌ Customer likely to Churn (Probability: {probability*100:.2f}%)"
        else:
            result = f"✅ Customer will Stay (Probability of churn: {probability*100:.2f}%)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
