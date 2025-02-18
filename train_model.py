import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

# Load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv('creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Handle imbalanced dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train and save the model
def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save the model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as model.pkl")

if __name__ == '__main__':
    train_and_save_model()