# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: CHAITANYA P S
RegisterNumber:  212222230024
*/

import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()

data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![327130452-666a2fbe-b1e9-4389-bf89-a54ee4fe1de3](https://github.com/chaitanya18c/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119392724/bc58452d-1241-4138-87bd-725f18161e9b)
![327130462-72448a19-ec6f-425c-8f14-34d4125032e1](https://github.com/chaitanya18c/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119392724/13a3e6da-2c61-4d8f-b095-f7ba0a2a7c97)
![327130481-5894ab20-ef10-45ee-91f1-f099cb3733da](https://github.com/chaitanya18c/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119392724/24656bb6-b943-4bc1-a50d-b9f858217719)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
