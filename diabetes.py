import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

diabetes_dataset = pd.read_csv('diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()

scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state = 2)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

#accuracy on training data
X_train_prediction = classifier.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Train Data Accuracy is", train_data_accuracy)

#accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Test Data Accuracy is", test_data_accuracy)

#Making a Predicting System
input_data = [4,110,92,0,0,37.6,0.191,30]

#input data as numpy array
input_data_as_np = np.asarray(input_data)

#reshape the array as we predicting only one instance
input_data_reshaped = input_data_as_np.reshape(1,-1)

#standardized the input data
std_data = scaler.transform(input_data_reshaped)

# print(std_data)

prediction = classifier.predict(std_data)
# print(prediction)

if(prediction[0]==0):
  print('Person is Not Diabetic')
else:
  print('Person is Diabetic')

joblib.dump(classifier,r"diabetes_model.pkl")