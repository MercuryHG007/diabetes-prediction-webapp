from flask import Flask, render_template, request
import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from diabetes import scaler

app = Flask(__name__)


model = joblib.load(r'diabetes_model.pkl')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()

@app.route("/predict", methods=['POST'])

def predict():
    if request.method == 'POST':
        # read data from the form
        input_data = request.form.to_dict()
        input_data = list(input_data.values())
        
        # input_data = [pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]

        #input data as numpy array
        input_data_as_np = np.asarray(input_data)

        #reshape the array as we predicting only one instance
        input_data_reshaped = input_data_as_np.reshape(1,-1)
        print(input_data_reshaped)

        #standardized the input data
        std_data = scaler.transform(input_data_reshaped)
        print(std_data)
        
        prediction=model.predict(std_data)
        print(prediction)
        output=prediction[0]
        
        if output==0:
            return render_template('result.html',prediction_text="You are Not Diabetic!")
        else:
            return render_template('result.html',prediction_text="You are Diabetic!")
    else:
        return render_template('result.html',prediction_text="Enter Correct DATA")


if __name__=="__main__":
    app.run(debug=True, port=5001)
