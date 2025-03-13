import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("Titanic-Dataset.csv")


print(df.info())  
print(df.head())  


df['Age'].fillna(df['Age'].median(), inplace=True) 
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  
df.drop(columns=['Cabin'], inplace=True) 


df.drop_duplicates(inplace=True)


Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]


df['Sex'] = df['Sex'].str.lower().replace({'male': 'Male', 'female': 'Female'})


df.to_csv("titanic_cleaned.csv", index=False)
print("Cleaned dataset saved as 'titanic_cleaned.csv'")


print(df.describe())  
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()


sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()


sns.pairplot(df, hue='Survived')
plt.show()
