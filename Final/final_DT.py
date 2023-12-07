#%%

import pandas as pd

file_path = 'data/bank.csv'
# Load the dataset
data = pd.read_csv(file_path)
# data.head()
# data.info()

#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encoding categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Splitting the dataset into training and testing sets
X = data.drop(['deposit', 'duration'], axis=1)
y = data['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#%%




#%%