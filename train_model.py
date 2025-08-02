import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
credit_card_data = pd.read_csv(r'D:\Creditcardfraudprediction\archive\creditcard.csv')  # Use raw string

# Preprocessing
new_dataset = credit_card_data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
X = new_dataset.drop(columns='Class', axis=1)  # Features
Y = new_dataset['Class']  # Target variable

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the model
joblib.dump(model, 'credit_card_fraud_model.pkl')

# Print accuracy on test data
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print(f'Accuracy on Test Data: {test_accuracy}')
