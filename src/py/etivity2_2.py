from sklearn.feature_selection import chi2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers_iqr(data, threshold=1.5):
    # Calculate the first and third quartiles
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Identify outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers

df = pd.read_csv("diabetes.csv")
print(df.head())

#IQR
outliers = detect_outliers_iqr(df['feature_name'])

# Filter the DataFrame to include only rows without outliers
df_no_outliers = df[~outliers]


#Replace outliers with NA
outliers = detect_outliers_iqr(df['feature_name'])
df.loc[outliers, 'feature_name'] = pd.NA

# Use dropna() to select rows without any missing values
df_complete_cases = df.dropna()

print(df_complete_cases)

#chi2
information_gain = ch2(X, y)
feature_ranking = sorted(zip(X.columns, information_gain), key=lambda x: x[1], reverse=True)

for feature, gain in feature_ranking:
    print(f"Feature: {feature}, Information Gain: {gain}")


#6th point in task

# Get the name of the feature with the highest information gain score
discriminating_feature = feature_ranking[0][0]

# Get the name of the feature with the smallest information gain score
non_discriminating_feature = feature_ranking[-1][0]

# Create boxplot and distribution plot for the discriminating feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=y, y=X[discriminating_feature])
plt.title(f"Boxplot of {discriminating_feature}")

plt.subplot(1, 2, 2)
sns.histplot(data=X, x=discriminating_feature, hue=y, kde=True)
plt.title(f"Distribution plot of {discriminating_feature}")

plt.tight_layout()
plt.show()

# Create boxplot and distribution plot for the non-discriminating feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=y, y=X[non_discriminating_feature])
plt.title(f"Boxplot of {non_discriminating_feature}")

plt.subplot(1, 2, 2)
sns.histplot(data=X, x=non_discriminating_feature, hue=y, kde=True)
plt.title(f"Distribution plot of {non_discriminating_feature}")

plt.tight_layout()
plt.show()
