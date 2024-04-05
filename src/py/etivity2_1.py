from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('food.csv')
print(df.head()) #.tail

#Dimensions
print("Dimensions of the dataset:", df.shape)

#missing values
print("Number of missing values:")
print(df.isnull().sum())

#Feature scaling???
print("Summary statistics for the features:")
print(df.describe())

#z-transform
scaler = StandardScaler()
num_columns = df.columns[1:] 

# Apply z-transformation to the numerical columns
df[num_columns] = scaler.fit_transform(df[num_columns]) #scaler.fit(df)

print(df.head())

#PCA
pca = PCA()

# Fit PCA to the preprocessed data
pca.fit(df[numerical_columns])

# Transform the data onto the principal components
pca_data = pca.transform(df[numerical_columns])

# Convert the transformed data to a DataFrame
pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])

# Concatenate the transformed data with the 'Country' column
pca_df = pd.concat([df['Country'], pca_df], axis=1)

# Display the first few rows of the PCA-transformed DataFrame
print(pca_df.head())


# Plot PC1 against PC2
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], color='blue', alpha=0.7)

# Annotate points with country names
for i, txt in enumerate(pca_df['Country']):
    plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=8)

# Add labels and title
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('Score Plot of Principal Components 1 vs 2')

# Show plot
plt.grid(True)
plt.show()

