from flask import Blueprint, request, jsonify
import pandas as pd
from .utils import to_numeric, predict
# from .ratio_utils import predict_ratio   # для TF-модели

bp = Blueprint("api", __name__)



# -------- свойства (sklearn) --------
@bp.route("/predict/modulus", methods=["POST"])
def predict_modulus():
    """
    Предсказание модуля упругости
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: data
        description: данные для расчета
        required: true
        schema:
          type: object
          properties:
            rows:
              type: array
              items:
                type: object
    responses:
      200:
        description: результаты предсказания
        examples:
          application/json: [{"Модуль упругости при растяжении, ГПа": 123.4}]
    """
    data = request.get_json(force=True)
    df = pd.DataFrame(data.get("rows", []))
    df = to_numeric(df)
    df = predict(df, "Модуль упругости при растяжении, ГПа")
    return jsonify(df.to_dict(orient="records"))

@bp.route("/predict/strength", methods=["POST"])
def predict_strength():
    """
    Предсказание прочности при растяжении
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: data
        description: данные для расчета
        required: true
        schema:
          type: object
          properties:
            rows:
              type: array
              items:
                type: object
    responses:
      200:
        description: результаты предсказания
        examples:
          application/json: [{"Прочность при растяжении, МПа": 456.7}]
    """
    data = request.get_json(force=True)
    df = pd.DataFrame(data.get("rows", []))
    df = to_numeric(df)
    df = predict(df, "Прочность при растяжении, МПа")
    return jsonify(df.to_dict(orient="records"))

# # -------- соотношение (TF) --------
# @bp.route("/predict/ratio", methods=["POST"])
# def predict_ratio_endpoint():
#     data = request.get_json(force=True)
#     df = pd.DataFrame(data.get("rows", []))
#     df = to_numeric(df)
#     df = predict_ratio(df)   # отдельная функция для ratio_model_tf
#     return jsonify(df.to_dict(orient="records"))