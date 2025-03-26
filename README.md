# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Srikaran M
RegisterNumber:212223040206
*/
```
```
import pandas as pd
df=pd.read_csv("Placement_Data.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/20fc6ab1-d4b0-4b99-b066-4f8e208b4750)

```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```
![image](https://github.com/user-attachments/assets/3175ba5d-43cc-48bf-bd04-bb635ccfcbbb)
```
d1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/2bda4122-e0ac-40d0-af6d-7bfb79ed28cc)
```
d1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/15a79e7f-3dc2-422d-9678-3191d7d3f5f0)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```
![image](https://github.com/user-attachments/assets/8fb0e4c3-b319-4512-9c4e-7bb500dff06c)
```
x=d1.iloc[:, : -1]
x
```
![image](https://github.com/user-attachments/assets/a095c719-6163-47cd-9a5f-bbec92eb44bd)
```
y=d1["status"]
y
```
![image](https://github.com/user-attachments/assets/dc08cc2e-de23-487a-9f28-d82a267eff79)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/b40fe15c-2655-4c87-a3a8-ef826ba69304)
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/53421f80-6719-4ef9-9ade-4760db45d381)
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/8b543b70-8d0c-45ea-8834-8263a959091c)
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/ac4ef3b9-4363-496a-876a-9070ff92493b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
