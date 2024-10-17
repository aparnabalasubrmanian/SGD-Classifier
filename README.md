# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
STEP 1: Start

STEP 2: Load the Iris dataset and create a Pandas DataFrame with features and target.

STEP 3: Split the dataset into features (X) and target (y).

STEP 4: Split the data into training and testing sets using train_test_split.

STEP 5: Initialize the SGDClassifier with default parameters.

STEP 6: Train the classifier using the training data.

STEP 7: Predict the target values for the testing set.

STEP 8: Calculate and display the model's accuracy and confusion matrix.

STEP 9: End
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: APARNA RB
RegisterNumber:  212222220005
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris=load_iris()

# Create a Pandas DataFrame
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

#Display the first few rows of the dataset
print(df.head())

#Split the data into features (X) and target (y)
X = df.drop('target',axis=1)
y=df['target']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create an SGD classifier with default parameters
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)

#Train the classifier on thr training data
sgd_clf.fit(X_train,y_train)

#Make predictions on the testing data
y_pred=sgd_clf.predict(X_test)

#Evaluate the classifier's accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

#Calculate the confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

```

## Output:

![368956354-5961f693-dbcc-4967-abdd-acd3f6e064d7](https://github.com/user-attachments/assets/f30dbd25-5642-403b-be42-8a126d45a0c3)

ACCURACY
![368956514-6041e366-e6a0-4477-9097-f71e9c8abf65](https://github.com/user-attachments/assets/652fd42e-849b-425f-949b-8bae18357a16)

![368956638-ca0b2ae5-6011-4a22-b838-d78badc1b604](https://github.com/user-attachments/assets/fb3205a7-f5c8-49b6-a0e2-0af79f9b3363)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
