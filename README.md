# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SUJITHRA B K N
RegisterNumber:  212222230153


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)



```

## Output:

### df.head()

![308935820-d64b9d18-7b39-422d-8967-cf750e4ad4c1](https://github.com/sujithrabkn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477857/443e6fcc-be33-4bd7-8a0f-32c1183572b3)

### Array value of X

![308936146-9962ab52-f6b7-4143-95fd-9479f34c73e3](https://github.com/sujithrabkn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477857/10298784-5036-4787-9eb0-f134b54792fd)

### Array value of Y

![308936384-728df3ad-a1d9-4945-a6f2-ca885220e96d](https://github.com/sujithrabkn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477857/971e721e-d0b0-41d8-bde9-77599a644bca)

### Values of y prediction

![308936824-ec9cd745-f12e-4044-beaa-8f78d48a7f11](https://github.com/sujithrabkn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477857/3a9e05fe-b1e2-43c8-897e-69ef6ea32192)

### Training set

![308939751-f854ae71-d3ac-4e40-bd14-25a20359553f](https://github.com/sujithrabkn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477857/f58692f1-62c7-4d80-91bb-d12eaa22a446)

### Values of MSE,MAE and RMSE

![308937838-b4e0fc8f-0e28-46b8-a346-4df723954496](https://github.com/sujithrabkn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477857/8e664c0d-ea71-471d-be05-d7bc892b733e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
