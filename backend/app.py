from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd

from preprocessing.preprocess import full_preprocessing

app = Flask(__name__)

# =========================
# PATH SETTINGS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER  = os.path.join(BASE_DIR, "model")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL + SCALER
# =========================
model  = joblib.load(os.path.join(MODEL_FOLDER, "trained_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))


# =========================
# SHARED HELPER: assign risk levels from anomaly scores
# =========================
def assign_risk(df):
    high_th = df["avg_anomaly_score"].quantile(0.2)
    med_th  = df["avg_anomaly_score"].quantile(0.5)

    def risk_level(row):
        s = row["avg_anomaly_score"]
        if s <= high_th:
            return "high"
        elif s <= med_th:
            return "medium"
        else:
            return "low"

    df["risk_level"] = df.apply(risk_level, axis=1)
    return df


# =========================
# SHARED HELPER: build explanations
#
# Logic:
#   - Reasons are derived from dataset-relative percentile thresholds
#     so they reflect how each consumer compares to the whole batch,
#     not arbitrary hardcoded cutoffs.
#   - The explanation tone then matches the risk level:
#       high   → serious anomaly language
#       medium → under monitoring language
#       low    → clean bill of health (no flagging of minor stats)
# =========================
def build_explanations(df):
    # Compute dataset-relative thresholds (worst 25% flagged per metric)
    peak_th  = (df["max_consumption"] / df["avg_consumption"]).quantile(0.75)
    var_th   = (df["variability"]     / df["avg_consumption"]).quantile(0.75)
    lf_th    = df["load_factor"].quantile(0.25)   # low load factor = suspicious

    explanations = []
    for _, row in df.iterrows():
        peak_ratio = row["max_consumption"] / row["avg_consumption"]
        var_ratio  = row["variability"]     / row["avg_consumption"]
        lf         = row["load_factor"]
        risk       = row["risk_level"]

        reasons = []
        if peak_ratio >= peak_th:
            reasons.append("unusually high peak-to-average consumption ratio")
        if var_ratio >= var_th:
            reasons.append("high variability relative to average usage")
        if lf <= lf_th:
            reasons.append("low load factor indicating irregular usage patterns")

        if risk == "high":
            if reasons:
                explanations.append(
                    "Likely theft or tampering — " + ", ".join(reasons) + "."
                )
            else:
                explanations.append(
                    "Consumption pattern significantly deviates from normal behavior."
                )

        elif risk == "medium":
            if reasons:
                explanations.append(
                    "Flagged for review — " + ", ".join(reasons) + "."
                )
            else:
                explanations.append(
                    "Consumption shows moderate deviation; under monitoring."
                )

        else:  # low
            explanations.append("Normal consumption pattern — no anomalies detected.")

    return explanations


# =========================
# LOAD INITIAL NOTEBOOK DATA
# =========================
try:
    csv_path = os.path.join(
        BASE_DIR,
        "..",
        "notebooks",
        "data",
        "processed",
        "explainable_anomalies.csv"
    )

    print("Loading:", csv_path)

    df = pd.read_csv(csv_path)

    print("Startup CSV columns:", df.columns.tolist())

    # Keep only raw feature columns so the model always scores fresh
    df = df[["consumer_id", "avg_consumption", "max_consumption", "variability", "load_factor"]]

    # Score with model
    X        = df[["avg_consumption", "max_consumption", "variability", "load_factor"]]
    X_scaled = scaler.transform(X)
    df["avg_anomaly_score"] = model.decision_function(X_scaled)
    df["anomaly_label"]     = model.predict(X_scaled)

    # Risk + explanations
    df = assign_risk(df)
    df["explanation"] = build_explanations(df)

    global_results = df.to_dict(orient="records")

    print("Sample explanation:", global_results[0].get("explanation") if global_results else "none")
    print("Loaded rows:", len(global_results))

except Exception as e:
    import traceback
    print("Load failed:", e)
    traceback.print_exc()
    global_results = []


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    total  = len(global_results)
    high   = len([x for x in global_results if x.get("risk_level") == "high"])
    medium = len([x for x in global_results if x.get("risk_level") == "medium"])
    low    = len([x for x in global_results if x.get("risk_level") == "low"])

    return render_template(
        "index.html",
        data=global_results,
        total=total,
        high=high,
        medium=medium,
        low=low
    )


# =========================
# CSV UPLOAD
# =========================
@app.route("/upload_csv", methods=["POST"])
def upload_csv():

    global global_results

    file     = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocessing
    processed_df = full_preprocessing(filepath)

    # Score
    X        = processed_df[["avg_consumption", "max_consumption", "variability", "load_factor"]]
    X_scaled = scaler.transform(X)
    processed_df["avg_anomaly_score"] = model.decision_function(X_scaled)
    processed_df["anomaly_label"]     = model.predict(X_scaled)

    # Risk + explanations
    processed_df = assign_risk(processed_df)
    processed_df["explanation"] = build_explanations(processed_df)

    # Update + save
    global_results = processed_df.to_dict(orient="records")
    processed_df.to_csv(
        os.path.join(UPLOAD_FOLDER, "latest_result.csv"),
        index=False
    )

    return jsonify({
        "message": "success",
        "rows_processed": len(processed_df)
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
