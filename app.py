import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('notebook/modelo.pkl','rb'))

@app.route("/")
def verifica_api_online():
  return "API ONLINE v1.0", 200

@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json(force=True)
  prediction = model.predict(np.array([list(data.values())]))
  output = prediction[0]

  resposta = {'DIABETES': int(output)}
  return jsonify(resposta)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)
