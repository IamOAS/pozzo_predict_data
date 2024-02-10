from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON request
        input_data = request.get_json()

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        predictions = model.predict(input_scaled)

        # Return the prediction as JSON
        return jsonify({'prediction': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
