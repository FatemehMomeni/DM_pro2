from sklearn.impute import SimpleImputer
import xlrd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier




pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

df = pd.read_excel("dataset.xls")

# fill missing values with column mean
cl_mean = df['Close_Value'].mean()
df['Close_Value'].fillna(cl_mean, inplace=True)

# find noisy data
df_sort = df.sort_values(by=['Close_Value'], ascending=True)
q1 = np.quantile(df['Close_Value'], 0.25)
q3 = np.quantile(df['Close_Value'], 0.75)
IQR = q3 - q1
lower_fence = q1 - (1.5 * IQR)
upper_fence = q3 + (1.5 * IQR)
for row in range(len(df['Close_Value'])):
    if df['Close_Value'][row] < lower_fence or df['Close_Value'][row] > upper_fence:
        df = df.drop(row, axis=0)

#print(df['Close_Value'])


correlation = pd.DataFrame(df, columns=['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID', 'Product',
                              'Close_Value', 'Created Date', 'Close Date'])
print(correlation.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
print("************************************")
#according to correlation [customer,ContactEmailID], [Agent,SalesAgentEmailID] has one value so we get just ContactEmailID & Agent

modify_dataset1 = pd.DataFrame(df, columns=['Agent', 'ContactEmailID', 'Product', 'Close_Value', 'Created Date', 'Close Date'])
modify_dataset2 = pd.DataFrame(df, columns=['Stage'])
# print(X)

# ###################convert categorical data to numeric##########################
X = modify_dataset1.apply(LabelEncoder().fit_transform)
Y = modify_dataset2.apply(LabelEncoder().fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
"""print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
#######################################################
"""""

# ############################## decision tree ##################################
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# ############################## KNN ##################################
print("************************************")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(X_test, y_test)))

# ############################## Naive bayes ##################################
print("************************************")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
      .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
      .format(gnb.score(X_test, y_test)))

# ############################## Random Forest ##################################
print("************************************")
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
print('Accuracy of Random Forest classifier on training set: {:.2f}'
      .format(classifier.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
      .format(classifier.score(X_test, y_test)))