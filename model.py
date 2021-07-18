import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#dataframe
train = pd.read_csv("train.csv")

#Fill categorical null values
def fill_categorical_null_values(df, variable):
    df[variable] = df[variable].fillna(df[variable].mode()[0])
    return df[variable]

train["Gender"] = fill_categorical_null_values(train, "Gender")
train["Married"] = fill_categorical_null_values(train, "Married")
train["Dependents"] = fill_categorical_null_values(train, "Dependents")
train["Self_Employed"] = fill_categorical_null_values(train, "Self_Employed")

#Fill numerical null values
def fill_numerical_null_values(df, variable):
    df[variable] = df[variable].fillna(df[variable].median())
    return df[variable]

train["LoanAmount"] = fill_numerical_null_values(train, "LoanAmount")
train["Loan_Amount_Term"] = fill_numerical_null_values(train, "Loan_Amount_Term")
train["Credit_History"] = fill_numerical_null_values(train, "Credit_History")
train["LoanAmount"] = fill_numerical_null_values(train, "LoanAmount")

#drop redundant features
train.drop(columns = ["Loan_ID", "Loan_Status"], inplace = True)

#one hot encoding
train = pd.get_dummies(train, drop_first = True)
train2 = train.copy()

#outliers detection
def outlier_caping(df, variable):
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return lower_limit, upper_limit

lower_limit, upper_limit = outlier_caping(train2, "ApplicantIncome")
train2['ApplicantIncome'] = np.where(train2['ApplicantIncome'] > upper_limit, upper_limit, train2['ApplicantIncome'])
train2['ApplicantIncome'] = np.where(train2['ApplicantIncome'] < lower_limit, lower_limit, train2['ApplicantIncome'])

lower_limit, upper_limit = outlier_caping(train2, "CoapplicantIncome")
train2['CoapplicantIncome'] = np.where(train2['CoapplicantIncome'] > upper_limit, upper_limit, train2['CoapplicantIncome'])
train2['CoapplicantIncome'] = np.where(train2['CoapplicantIncome'] < lower_limit, lower_limit, train2['CoapplicantIncome'])

lower_limit, upper_limit = outlier_caping(train2, "LoanAmount")
train2['LoanAmount'] = np.where(train2['LoanAmount'] > upper_limit, upper_limit, train2['LoanAmount'])
train2['LoanAmount'] = np.where(train2['LoanAmount'] < lower_limit, lower_limit, train2['LoanAmount'])

#final dataset
train_final = train2.copy()

#dependent and independent features
X, y  = train_final.drop(columns =  ["LoanAmount"], axis = 1), train["LoanAmount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
regressor = RandomForestRegressor(n_estimators=200, min_samples_split=100, min_samples_leaf=5,max_features='auto',max_depth=20)
model = regressor.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred[:10])
params = [3434.5, 2700.0, 3500.0, 1.0, 0, 1, 1, 0, 0, 1, 0, 1, 0]
act_params = np.array(params)

#load model
pickle.dump(model, open('model_loan.pkl', 'wb'))

#load the model
model = pickle.load(open('model_loan.pkl', 'rb'))
print(model.predict([act_params]))
