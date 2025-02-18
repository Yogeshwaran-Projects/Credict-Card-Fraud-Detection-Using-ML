import numpy as np
import pickle

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Make a prediction
def predict(model, input_data):
    prediction = model.predict(input_data)
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

# Main function
def main():
    # Load the model
    model = load_model()

    # Predefined inputs
    fraud_input = [1.0, -1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0]
    not_fraud_input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]

    # Reshape inputs for the model
    fraud_input = np.array(fraud_input).reshape(1, -1)
    not_fraud_input = np.array(not_fraud_input).reshape(1, -1)

    # Make predictions
    fraud_result = predict(model, fraud_input)
    not_fraud_result = predict(model, not_fraud_input)

    # Display results
    print("Test Case 1 (Fraud):")
    print(f"Input: {fraud_input}")
    print(f"Prediction: {fraud_result}\n")

    print("Test Case 2 (Not Fraud):")
    print(f"Input: {not_fraud_input}")
    print(f"Prediction: {not_fraud_result}")

if __name__ == '__main__':
    main()