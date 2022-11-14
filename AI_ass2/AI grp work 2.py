#install of  pandas library installed. Then import the library.
import inline as inline
import matplotlib
import sklearn
import pandas as pd #data manipulation library that is necessary for every aspect of data analysis or machine learning.

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from numpy import block

# file is in same path as python script

path_to_file = r"C:\Users\zinda\PycharmProjects\pythonProject\AI_ass2\AI_dataset.csv"


AI_data = pd.read_csv(path_to_file)

AI_data['Gender'].unique()


#Exploratory Data Analysis
AI_data.shape #returns the orientation of the dataset i.e number of columns and rows.

#Show first 5 rows by default. Change by adding number into parentheses.
print(AI_data.head())

print(AI_data.describe()) #returns some statistical information for the data


#Number of rows belonging to each class
print(AI_data.groupby('Purchased').size())

#The dataset contain five columns: User Id, Age, Gender, EstimatedSalary  , Purchased.

#There is need to split data into two arrays: X (features) and y (labels).

feature_columns = [ 'Age', 'EstimatedSalary']
X = AI_data[feature_columns].values
y = AI_data['Purchased'].values

#The labels here are categorical. The KNeighborsClassifier does not accept string labels. We need to use LabelEncoder to transform them into numbers.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)





#Spliting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Data Visualization Libraries
#import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

plt.figure()
AI_data.drop("User ID", axis=1).boxplot(by="Purchased", figsize=(10, 5))
plt.show()






#KNN Predicions
# Fitting clasifier to the Training set
# Loading libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 5)
classifier = KNeighborsClassifier(n_neighbors=5)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Model Evaluation
#Building a Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm




#Calculating Model Accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

#Predict Output

x1=float(input("Enter Age:"))
x2=float(input("Enter Salary:"))


predicted_class= classifier.predict([[x1,x2]])#x1,x2, are the different values entered by users for the features
#print(predicted)
if predicted_class==0:
    print("...The customer will not purchase a new car...")
elif predicted_class==1:
     print("...The customer will  purchase a new car...")

else:
    print("...Class out of Range...")


