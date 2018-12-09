import numpy as np
import matplotlib.pyplot as plt
import pandas as pdb

# Importing dataset
dataset = pd.read_csv('FileName')
X = dataset.iloc[:, :].values
y = dataset.iloc[:].values

# Taking Care of Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit("Column")
X[:, :] = imputer.transform(X[:, :])

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0])
oneHotEncode = OneHotEncoder()
oneHotEncode = OneHotEncoder(categorical_features=[0])
X = oneHotEncode.fit_transform(X).toarray()

# Splitting the dataset
from sklearn.cross_validation import train_test_split
X_test, X_train, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
