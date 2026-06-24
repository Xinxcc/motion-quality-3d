from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# get the featues from mDTW
path_feature_sensor_X = r"F:\studium\Masterarbeit\X\Data\sensor_features.csv"
path_feature_camera_X = r"F:\studium\Masterarbeit\X\Data\camera_features.csv"

path_feature_sensor_S = r"F:\studium\Masterarbeit\S\Data\sensor_features.csv"
path_feature_camera_S = r"F:\studium\Masterarbeit\S\Data\camera_features.csv"

path_feature_sensor_P = r"F:\studium\Masterarbeit\P\Data\sensor_features.csv"
path_feature_camera_P = r"F:\studium\Masterarbeit\P\Data\camera_features.csv"

df1 = pd.read_csv(path_feature_sensor_X)
df1 = df1.drop(columns=df1.columns[0], axis=1)
df2 = pd.read_csv(path_feature_camera_X)
df2 = df2.drop(columns=df2.columns[0], axis=1)

data_sensor_X = df1.iloc[:, :-1]
lable_sensor_X = df1.iloc[:, -1]
data_camera_X = df2.iloc[:, :-1]
lable_camera_X = df2.iloc[:, -1]


df3 = pd.read_csv(path_feature_sensor_S)
df3 = df3.drop(columns=df3.columns[0], axis=1)
df4 = pd.read_csv(path_feature_camera_S)
df4 = df4.drop(columns=df4.columns[0], axis=1)

data_sensor_S = df3.iloc[:, :-1]
lable_sensor_S = df3.iloc[:, -1]
data_camera_S = df4.iloc[:, :-1]
lable_camera_S = df4.iloc[:, -1]

df5 = pd.read_csv(path_feature_sensor_P)
df5 = df5.drop(columns=df5.columns[0], axis=1)
df6 = pd.read_csv(path_feature_camera_P)
df6 = df6.drop(columns=df6.columns[0], axis=1)

data_sensor_P = df5.iloc[:, :-1]
lable_sensor_P = df6.iloc[:, -1]
data_camera_P = df5.iloc[:, :-1]
lable_camera_P = df6.iloc[:, -1]


df_camera = pd.concat([df2, df4, df6], axis=0)
df_sensor = pd.concat([df1, df3, df5], axis=0)

lable_camera = pd.concat([lable_camera_X, lable_camera_S, lable_camera_P], axis=0)
lable_sensor = pd.concat([lable_sensor_X, lable_sensor_S, lable_sensor_P], axis=0)

data_camera = df_camera.iloc[:, :-1]
data_sensor = df_sensor.iloc[:, :-1]

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_sensor_P,lable_sensor_P,test_size=0.2)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_camera_X,lable_camera_X,test_size=0.2)

Xtrain = data_sensor_P
Ytrain = lable_sensor_P
Xtest = data_camera_S
Ytest = lable_camera_S

# # choose the best n_estimator
# param_grid = {'n_estimators': [50, 100, 150, 200, 250]}

# # Create the Random Forests classifier
# rf = RandomForestClassifier(random_state=42)

# # Perform GridSearchCV with 5-fold cross-validation
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(Xtrain, Ytrain)

# # Get the best n_estimators value
# best_n_estimators = grid_search.best_params_['n_estimators']
# print("Best n_estimators:", best_n_estimators)


# train classifier

clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(n_estimators=50,random_state=0)
lin = LogisticRegression()


clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)
lin = lin.fit(Xtrain, Ytrain)

y= rfc.predict(Xtest)


cm = confusion_matrix(Ytest, y)

# Create a DataFrame from the confusion matrix for better visualization
class_names = np.unique(y)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


accuracy = accuracy_score(Ytest, y)
score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)
score_l = lin.score(Xtest, Ytest)


print("Single Tree:{}".format(score_c), "Random Forest:{}".format(score_r),"LogisticRegression:{}".format(accuracy),y)
# #print(lin.intercept_,lin.coef_)