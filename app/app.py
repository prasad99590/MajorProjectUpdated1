from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

from flask import Flask, render_template, request
import google.generativeai as genai
import os
import re


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = Flask(__name__)

def get_gemini_response(input_data):
    prompt_text = f"""
    Weight Category: {input_data['category']}, Vegetarian: {input_data['veg_or_nonveg']}, Disease: {input_data['disease']}, Region: {input_data['region']}, 
    Allergics: {input_data['allergics']}. 
    Please recommend 5 breakfast ideas,5 lunch ideas, 5 dinner ideas, and 6 workout routines.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro") 
    response = model.generate_content(prompt_text)
    print(response.text)
    return response.text

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    age = int(request.form['age'])
    gender = request.form['gender']  
    medical_history = request.form['medical_history']  

    gender_encoder = LabelEncoder()
    medical_history_encoder = LabelEncoder()

    gender_encoder.fit(["Male", "Female"])
    medical_history_encoder.fit(["Diabetes", "High BP","Low BP","None"])

    
    gender_encoded = gender_encoder.transform([gender])[0]  # Encode gender
    medical_history_encoded = medical_history_encoder.transform([medical_history])[0]  # Encode medical history

   
    new_sample = np.array([height, weight, age, gender_encoded, medical_history_encoded])

    # Step 4: Reshape to match model input shape (1 sample with n features)
    new_sample_reshaped = new_sample.reshape(1, -1)
    prediction = model.predict(new_sample_reshaped)
    print(prediction)
    # Return the result
    ans = ""

    match prediction:
        case 0:
            ans = "Normal"
        case 1:
            ans = "Obese"
        case 2:
            ans = "Over Weight"
        case 3:
            ans = "Under Weight"

    return render_template('category.html',data={"category":ans})



@app.route('/getform', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    category = request.form['category']
    veg_or_noveg = request.form['veg_or_nonveg']
    disease = request.form['disease']
    region = request.form['region']
    allergics = request.form['allergics']
    input_data = {
        'category' : category,
        'veg_or_nonveg': veg_or_noveg,
        'disease': disease,
        'region': region,
        'allergics': allergics
    }
    response=get_gemini_response(input_data)
    print(type(response))
    data={
        'content':response
    }
    return render_template('result.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)
