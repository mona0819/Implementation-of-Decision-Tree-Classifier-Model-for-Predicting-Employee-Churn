# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load employee data and split it into training and testing sets.
2. Train a Decision Tree classifier using entropy as the split criterion.
3. Evaluate the model using accuracy, confusion matrix, and classification report.
4. .Use the trained model to predict whether a new employee will stay or leave.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mohana Priya D
RegisterNumber:  212225230182
*/
```
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


data = pd.read_csv("Employee.csv")

data = pd.get_dummies(data, drop_first=True)


X = data.iloc[:, :-1]   # All columns except last
y = data.iloc[:, -1]    # Last column as target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier(criterion='gini', random_state=42)

model.fit(X_train, y_train)


plt.figure(figsize=(25,12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(i) for i in model.classes_],
    filled=True
)

plt.title("Decision Tree Classifier")
plt.show()
```
## Output:
<img width="1277" height="623" alt="image" src="https://github.com/user-attachments/assets/ce7deadb-bf8c-4a01-998a-ba1e87dfeb29" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
