from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Trained ML model load
model = joblib.load("churn_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    tenure = int(request.form["tenure"])
    monthly_charges = float(request.form["monthly_charges"])
    total_charges = float(request.form["total_charges"])
    contract = request.form["contract"]
    internet_service = request.form["internet_service"]
    payment_method = request.form["payment_method"]
    support_calls = int(request.form["support_calls"])
    tech_support = request.form["tech_support"]

    customer = pd.DataFrame([{
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "support_calls": support_calls,
        "tech_support": tech_support
    }])

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]
    probability = round(probability * 100, 2)

    if prediction == 1:
        result = f"🔴 High Churn Risk — {probability}%"
    else:
        result = f"🟢 Low Churn Risk — {probability}%"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )