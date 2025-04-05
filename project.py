import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Vaishnavi B M\Desktop\hackathon\heart.csv")

# Basic dataset overview
print("First 5 rows of the dataset:")
print(dataset.head())
print("Last 5 rows of the dataset:")
print(dataset.tail())

# Shape and info
print(f"Number of rows: {dataset.shape[0]}")
print(f"Number of columns: {dataset.shape[1]}")
print(dataset.info())

# Checking for missing values and duplicates
print("\nMissing values per column:")
print(dataset.isnull().sum())

print("\nAre there duplicates?:", dataset.duplicated().any())
dataset = dataset.drop_duplicates()
print(f"Shape after removing duplicates: {dataset.shape}")

# Descriptive statistics
print("\nSummary statistics:")
print(dataset.describe())

# Correlation matrix
print("\nCorrelation matrix:")
print(dataset.corr())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Target value counts
print("\nTarget value counts:")
print(dataset['target'].value_counts())

sns.countplot(x='target', data=dataset, palette="Set2")
plt.title("Count of Target Values")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# Sex distribution
print("\nSex distribution:")
print(dataset['sex'].value_counts())

sns.countplot(x='sex', data=dataset, palette="Set2")
plt.xticks([0, 1], ['Female', 'Male'])
plt.title("Sex Distribution")
plt.show()

# Sex vs Target
sns.countplot(x='sex', hue='target', data=dataset, palette="Set2")
plt.xticks([0, 1], ['Female', 'Male'])
plt.legend(labels=['No Disease', 'Disease'])
plt.title("Sex vs Target")
plt.show()

# Age distribution
sns.histplot(dataset['age'], bins=20, kde=True, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Chest pain types (cp)
sns.countplot(x='cp', data=dataset, palette="Set2")
plt.xticks([0, 1, 2, 3], ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], rotation=45)
plt.title("Chest Pain Types Distribution")
plt.show()

# Chest pain vs Target
sns.countplot(x='cp', hue='target', data=dataset, palette="Set2")
plt.xticks([0, 1, 2, 3], ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], rotation=45)
plt.legend(labels=["No Disease", "Disease"])
plt.title("Chest Pain Types vs Target")
plt.show()

# Fasting blood sugar (fbs) vs Target
sns.countplot(x='fbs', hue='target', data=dataset, palette="Set2")
plt.legend(labels=["No Disease", "Disease"])
plt.title("Fasting Blood Sugar vs Target")
plt.show()

# Blood pressure distribution
plt.figure(figsize=(6, 4))
sns.histplot(dataset['trestbps'], bins=20, kde=True, color='orange')
plt.title("Resting Blood Pressure Distribution")
plt.xlabel("Resting Blood Pressure")
plt.ylabel("Frequency")
plt.show()

# Cholesterol distribution
plt.figure(figsize=(6, 4))
sns.histplot(dataset['chol'], bins=20, kde=True, color='green')
plt.title("Cholesterol Level Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()

# Split columns into categorical and continuous
cate_val = []
cont_val = []

for column in dataset.columns:
    if dataset[column].nunique() <= 10:
        cate_val.append(column)
    else:
        cont_val.append(column)

print("\nCategorical variables:", cate_val)
print("Continuous variables:", cont_val)

# Histograms for continuous variables
for column in cont_val:
    plt.figure(figsize=(6, 4))
    plt.hist(dataset[column], bins=20, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

