from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
#load model
model = pickle.load(open('model_loan.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender_male = 0
    married_yes = 0
    dependents_1 = 0
    dependents_2 = 0
    dependents_3 = 0
    education_not_graduate =  0
    property_area_urban = 0
    property_area_semi_urban = 0
    self_employed_yes = 0
    if request.method == 'POST':
        #gender
        gender = request.form['Gender']
        if gender == "Male":
            gender_male = 1
        else:
            gender_male = 0
        
        #married
        married = request.form['Married']
        if married == "Yes":
            married_yes = 1
        else:
            married_yes = 0

        #dependents
        dependents = request.form['Dependents']
        if dependents == "1":
            dependents_1 = 1
            dependents_2 = 0
            dependents_3 = 0
        elif dependents == "2":
            dependents_1 = 0
            dependents_2 = 1
            dependents_3 = 0
        elif dependents == "3+":
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 1
        else:
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 0
        
        #education
        education = request.form['Education']
        if education == 'Graduate':
            education_not_graduate = 0
        else:
            education_not_graduate = 1
        
        #propertyarea
        property_area = request.form['Property Area']
        if property_area == "Urban":
            property_area_urban= 1
            property_area_semi_urban = 0
        elif property_area == "Semi Urban":
            property_area_urban= 0
            property_area_semi_urban = 1
        else:
            property_area_urban= 0
            property_area_semi_urban = 0

        #self employed
        self_employed = request.form['Self Employed']
        if self_employed == "Yes":
            self_employed_yes = 1
        else:
            self_employed_yes = 0

        #credit History
        credit_history = float(request.form['Credit History'])

        #Applicant income
        appplicantincome = float(request.form['ApplicantIncome'])

        #Applicant income
        coappplicantincome = float(request.form['CoApplicantIncome'])

        #Loan Amount Term
        loanamountterm = float(request.form['LoanAmountTerm'])

        params = [appplicantincome, coappplicantincome, loanamountterm, credit_history, gender_male, married_yes, dependents_1, dependents_2, dependents_3, education_not_graduate, self_employed_yes, property_area_semi_urban, property_area_urban]
        act_params = np.array(params)

        #prediction 
        pred = model.predict([act_params])
        output = pred[0]
        return render_template('index.html', prediction_text = 'You can take '+ str(output) + " Lakhs")

        



if __name__ == '__main__':
    app.run(debug = True)