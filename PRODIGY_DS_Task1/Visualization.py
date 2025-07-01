import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv', skiprows=4)

print(data.head())

population_data = data[['Country Name', '2022']]
population_data.columns = ['Country', 'Population_2022']

population_data.dropna(inplace=True)

population_data_sorted = population_data.sort_values(by='Population_2022', ascending=False)

print(population_data_sorted.head(10))
plt.figure(figsize=(12, 6))
sns.barplot(data=population_data_sorted.head(10), x='Country', y='Population_2022', palette='viridis')

plt.title('Top 10 Countries by Population in 2022')
plt.xticks(rotation=45)
plt.ylabel('Population')
plt.xlabel('Country')
plt.tight_layout()
plt.show()

