from sklearn.impute import SimpleImputer
import xlrd
import pandas as pd
import numpy as np

df = pd.read_excel("dataset.csv")

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

print(df['Close_Value'])
