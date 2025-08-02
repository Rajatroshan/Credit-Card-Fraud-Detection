from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('credit_card_fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from request
    input_features = np.array(data['features']).reshape(1, -1)  # Reshape input

    # Check the number of features
    if input_features.shape[1] != 30:  # Adjust according to your model's expected features
        return jsonify({"error": "Expected 30 features."}), 400

    prediction = model.predict(input_features)
    return jsonify({'prediction': int(prediction[0])})  # Return prediction

if __name__ == '__main__':
    app.run(debug=True)
