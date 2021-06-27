import xlrd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)


def dataset_divide(dataframe):
    # create dataframes as input of algorithms
    # according to correlation [Agent,SalesAgentEmailID] has same values so we just get Agent
    #X = pd.DataFrame(dataframe, columns=['Agent', 'Product', 'Close_Value', 'Created Date', 'Close Date'])
    X = pd.DataFrame(dataframe, columns=['Agent', 'Product', 'Close_Value', 'Created year', 'Created month',
                                         'Created day', 'Close year', 'Close month', 'Close day'])
    Y = pd.DataFrame(dataframe, columns=['Stage'])
    Y = np.ravel(Y)

    # divide datafrsm to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def output(model, random_farest):
    model.fit(X_train, y_train)
    print('Accuracy of classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
    if random_farest:
        print('Accuracy of classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
        print('F1score of classifier on test set: {:.2f}'.format(f1_score(y_test, rf.predict(X_test), average='macro')))
        print(confusion_matrix(y_test, rf.predict(X_test)))
    print('\n')


df = pd.read_excel("dataset.xls")
df['Created year'] = df['Created Date'].dt.year
df['Created month'] = df['Created Date'].dt.month
df['Created day'] = df['Created Date'].dt.day
df['Close year'] = df['Close Date'].dt.year
df['Close month'] = df['Close Date'].dt.month
df['Close day'] = df['Close Date'].dt.day
df.pop('Created Date')
df.pop('Close Date')


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

#no_outlier_df = pd.DataFrame(columns=['Agent', 'Stage', 'Product', 'Close_Value', 'Created Date', 'Close Date'])
no_outlier_df = pd.DataFrame(columns=['Agent', 'Stage', 'Product', 'Close_Value', 'Created year', 'Created month',
                                      'Created day', 'Close year', 'Close month', 'Close day'])
for row in range(len(df['Close_Value'])):
    if not(df.loc[row, 'Close_Value'] < lower_fence or df.loc[row, 'Close_Value'] > upper_fence):
        no_outlier_df = no_outlier_df.append(df.loc[row], ignore_index=True)


# convert categorical data to numeric
convert_numeric = no_outlier_df.apply(LabelEncoder().fit_transform)

# calculate correlation between attributes
correlation = convert_numeric.corr(method='pearson', min_periods=1)

# separate in_progress and won_lost stages
lost_df = pd.DataFrame(convert_numeric[convert_numeric.Stage == 1])
won_df = pd.DataFrame(convert_numeric[convert_numeric.Stage == 2])
no_inProgress_df = pd.concat([lost_df, won_df])

inProgress_df = convert_numeric[convert_numeric.Stage == 0]
#X_inProgress = pd.DataFrame(inProgress_df, columns=['Agent', 'Product', 'Close_Value', 'Created Date', 'Close Date'])
X_inProgress = pd.DataFrame(inProgress_df, columns=['Agent', 'Product', 'Close_Value', 'Created year', 'Created month',
                                                    'Created day', 'Close year', 'Close month', 'Close day'])

X_train, X_test, y_train, y_test = dataset_divide(no_inProgress_df)
dataset_divide(no_inProgress_df)


# decision tree
dt = DecisionTreeClassifier()
print("Decision Tree:")
output(dt, False)

# KNN
knn = KNeighborsClassifier()
print("KNN:")
output(knn, False)

# Naive bayes
gnb = GaussianNB()
print("Naive Bayes:")
output(gnb, False)

# Random Forest
rf = RandomForestClassifier(n_estimators=50)
print("Random Forest:")
output(rf, True)


y_predict = rf.predict(X_inProgress)

for row in range(len(convert_numeric)):
    if convert_numeric.loc[row, 'Stage'] == 0:
        convert_numeric.loc[row, 'Stage'] = y_predict[0]
        np.delete(y_predict, 0)

X_train, X_test, y_train, y_test = dataset_divide(convert_numeric)
dataset_divide(convert_numeric)

print("Random Forest:")
output(rf, True)
