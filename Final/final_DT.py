#%%

import pandas as pd

file_path = 'data/bank.csv'
# Load the dataset
data = pd.read_csv(file_path)
# data.head()
# data.info()

#%%

# Assuming 'data' is your DataFrame
poutcome_one_hot = pd.get_dummies(data['poutcome'], prefix='poutcome')

# To see the result
print(poutcome_one_hot.head())

#%%

# Concatenate the one-hot encoded columns to the original DataFrame
data = pd.concat([data, poutcome_one_hot], axis=1)

# Drop the original 'poutcome' column
data.drop('poutcome', axis=1, inplace=True)

# To see the updated DataFrame
print(data.head())



#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encoding categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Splitting the dataset into training and testing sets
#X = data.drop('deposit', axis=1)
X = data.drop(['deposit', 'duration'], axis=1)
y = data['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#%%

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Building the decision tree model
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

# Making predictions on the test data
y_pred = tree_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(accuracy)
print(classification_rep)

#%%

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Setting the figure size for better visibility
plt.figure(figsize=(20,10))

# Plotting the decision tree
plot_tree(tree_classifier, filled=True, feature_names=X.columns, class_names=['No Deposit', 'Deposit'], max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization (First 3 Levels)")

image_path = 'results/decision_tree_visualization.png'
plt.savefig(image_path)
#plt.show()

#%%

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Setting the figure size for better visibility
plt.figure(figsize=(20,10))

# Plotting the decision tree
plot_tree(tree_classifier, filled=True, feature_names=X.columns, class_names=['No Deposit', 'Deposit'], max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization (First 3 Levels)")

image_path = 'results/decision_tree_no_duration.png'
plt.savefig(image_path)


#%%