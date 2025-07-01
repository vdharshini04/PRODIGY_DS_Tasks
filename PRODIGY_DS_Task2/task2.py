import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
print(df.head())

print(df.isnull().sum())

# Fill or drop missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()
sns.countplot(x='Survived', hue='Sex_male', data=df)
plt.title('Survival by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.legend(['Female', 'Male'])
plt.show()
