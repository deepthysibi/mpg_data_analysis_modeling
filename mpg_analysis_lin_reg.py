#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#reading data
data=pd.read_csv("auto-mpg.csv")
data

df=data.copy()

# checking is there any null value
df.isnull().sum()

df.info()

# to get statistical information
df.describe()
df.describe().transpose()

#here data type of horsepower is given as object,so we should convert it
df[df.horsepower.str.isdigit()==False]

data.iloc[32]

#we get some unwanted rows in horsepower,so we replace it
df['horsepower']=df['horsepower'].replace('?',np.nan)
df.info()

df.isnull().sum()
#we get some nan value there so we fill it by median
df['horsepower'].median()
df['horsepower']=df['horsepower'].fillna(df['horsepower'].median())

df.info()
type(df['horsepower'][1])
df['horsepower']=df['horsepower'].astype(float)
df.info()

#ploting histogram to check whether data follows normal distribution or not
df.hist(figsize=(20,15))

#ploting distplot
plt.figure(figsize=(10,10))
plt.subplot(3,3,1)
sns.distplot(df['cylinders'],color="green")

plt.subplot(3,3,2)
sns.distplot(df['displacement'],color="green")


plt.subplot(3,3,3)
sns.distplot(df['horsepower'],color="green")
plt.subplot(3,3,4)
sns.distplot(df['weight'],color="green")

plt.subplot(3,3,5)
sns.distplot(df['acceleration'],color="green")

plt.subplot(3,3,6)
sns.distplot(df['model year'],color="green")


# ploting boxplot to find outliers
plt.figure(figsize=(20,15))
plt.subplot(3,3,1)
sns.boxplot(x=df['cylinders'],color="lightgreen")

plt.subplot(3,3,2)
sns.boxplot(x=df['displacement'],color="green")


plt.subplot(3,3,3)
sns.boxplot(x=df['horsepower'],color="green")
plt.subplot(3,3,4)
sns.boxplot(x=df['weight'],color="green")

plt.subplot(3,3,5)
sns.boxplot(x=df['acceleration'],color="green")

plt.subplot(3,3,6)
sns.boxplot(x=df['model year'],color="green")

plt.subplot(3,3,7)
sns.boxplot(x=df['origin'],color="green")


# handle outliers in horsepower and acceleration
df['horsepower']=df['horsepower'].clip(lower=df['horsepower'].quantile(0.05),upper=df['horsepower'].quantile(0.95))
df['acceleration']=df['acceleration'].clip(lower=df['acceleration'].quantile(0.05),upper=df['acceleration'].quantile(0.95))

sns.boxplot(df['horsepower'],color='lightgreen')
# we removed outliers

df.shape

# checking correlation between features
df.corr()
sns.heatmap(df.corr(),annot=True)

# by evaluating corr matrix we understand that weight and horsepower displacement are highly correlated
# so we remove it
import numpy as np
corrMatrix=df.corr().abs()

upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.90
corrFutures = [column for column in upperMatrix.columns if any(upperMatrix[column] > 0.90)]

df.drop(columns=corrFutures)


# training the model using linear regression
x=df.drop(['mpg','car name'],axis=1)
y=df['mpg']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)

y_train_pred=model.predict(x_train)
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)
