# Importing the Important Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Data PReprocessing Libraries
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inmporting Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Importing the Dataset
df = pd.read_csv('train.csv')

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

# Since Passenger ID, Name and Ticket are irrelevant columns so we will drop them
df = df.drop('PassengerId', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)

# Storing all Columns Names in a list
col_names = list(df.columns)

# Dropping Columns with High Amount of Missing Values
for col in col_names:
    if (df[col].isnull().sum() / df[col].shape[0]) > 0.3:
        df = df.drop(col, axis=1)

# Setting Up Encoding Strategy
enc = {
    'Embarked' : {'S' : 0, 'C' : 1, 'Q' : 2},
    'Sex' : {'male' : 0, 'female' : 1}
}

'''
# To Encode Embarked Column with One Hot Encoding Mechanism we can do this
embark = pd.get_dummies(df['Embarked'], drop_first=True)
df = df.drop('Embarked', axis=1)
df = pd.concat([df, embark], axis=1)
'''

# Encoding the Categorical Values
df = df.replace(enc)

# Separating Features and Class
X = df.iloc[:, 1:].values
y = df.iloc[:, 0:1].values

# Creating Instance of Imputer class to manage Missing Values
imputer_mode = Imputer(missing_values='NaN', strategy='most_frequent')
imputer_median = Imputer(missing_values='NaN', strategy='median')

# Fitting the Imputer and Managing Missing Values present in the Dataset
X[:, 6:] = imputer_mode.fit_transform(X[:, 6:])
X[:, 2:3] = imputer_median.fit_transform(X[:, 2:3])

# Splitting Dataset into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

#----------------------------------------Model Building and Training----------------------------------------

# Listing all the Classifiers and their respective Accuracy Scores
classifiers = ['Decision Tree Classifier', 'K-Nearest Neighbor Classifier', 'Random Forest Classifier', 'Logistic Regression']
scores = list()

# Training with Decision Tree Classifier
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training with K Nearest Neighbor Classifier
clf2 = KNeighborsClassifier(n_neighbors=9)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training with Random Forest Classifier
clf3 = RandomForestClassifier(n_estimators=20)
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training with Logistic Regression
clf4 = LogisticRegression()
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

#----------------------------------------Model Building and Training----------------------------------------

#----------------------------------------Model Evaluation----------------------------------------

# Evaluating Performance of all the Classifiers
sns.barplot(x=scores, y=classifiers)
plt.xlabel('Accuracy Score')
plt.ylabel('Classifier')
plt.title('Classifier Performance')
plt.show()

# As we can see that Logistic Regression has the best Accuracy Score, therefore we'll use it as the Final Model

#----------------------------------------Model Evaluation----------------------------------------

# Checking on Sample Data
ds = [
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    [1, 'male', 40, 2, 1, 363272, 'Q']
]

ds = pd.DataFrame(ds[1:], columns=ds[0])
ds = ds.replace(enc)

[print('Yes') if clf4.predict(ds)[0] == 0 else print('No')]