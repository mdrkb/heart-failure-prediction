import os
import pickle
from flask import Flask, request, jsonify
import xgboost as xgb


model_file = "{}/model.bin".format(os.path.dirname(os.path.realpath(__file__)))

with open(model_file, "rb") as mf:
    model, dv = pickle.load(mf)

app = Flask("heart_failure")


@app.route("/")
def home():
    return "Heart Failure Prediction"


@app.route("/predict", methods=["POST"])
def predict():
    record = request.get_json()

    X = dv.transform(record)
    features = dv.get_feature_names_out()
    dX = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dX)

    response = []
    for idx, pred in enumerate(y_pred):
        result = pred >= 0.4

        if result:
            probability = round(pred * 100, 3)
        else:
            probability = round((1 - pred) * 100, 3)

        response.append(
            {
                "id": idx,
                "heart_failure_probability": "{}%".format(probability),
                "heart_failure": bool(result),
            }
        )

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
