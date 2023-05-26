import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the first dataset into a Pandas DataFrame, specifying the separator as ','
df1 = pd.read_csv("AdAccount1(Masks).csv", sep=',')
# Load the second dataset into a Pandas DataFrame, specifying the separator as ','
df2 = pd.read_csv("AdAccount2(Blankets).csv", sep=',')

# Select all columns except the first column as features for the first dataset
X1 = df1.iloc[:, 1:]
# Select all columns except the first column as features for the second dataset
X2 = df2.iloc[:, 1:]

# Select the column to use as the target for the first dataset
y1 = df1['Results']
# Select the column to use as the target for the second dataset
y2 = df2['Results']

# Apply StandardScaler to the numerical features for the first dataset
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

# Apply StandardScaler to the numerical features for the second dataset
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# Create a linear regression instance for the first dataset
regressor1 = LinearRegression()
# Create a linear regression instance for the second dataset
regressor2 = LinearRegression()

# Use RFE algorithm to select all features for the first dataset
rfe1 = RFE(regressor1, n_features_to_select=len(X1.columns))
rfe1 = rfe1.fit(X1_scaled, y1)
# Use RFE algorithm to select all features for the second dataset
rfe2 = RFE(regressor2, n_features_to_select=len(X2.columns))
rfe2 = rfe2.fit(X2_scaled, y2)

# Get the names of the selected features for the first dataset
selected_features1 = X1.columns[rfe1.support_].tolist()
# Get the names of the selected features for the second dataset
selected_features2 = X2.columns[rfe2.support_].tolist()

# Add the "Results" column to the list of selected features for the first dataset
selected_features1.append('Results')
# Add the "Results" column to the list of selected features for the second dataset
selected_features2.append('Results')

# Create a subset of the selected features and the "Results" column for the first dataset
X_subset1 = df1[selected_features1]
# Create a subset of the selected features and the "Results" column for the second dataset
X_subset2 = df2[selected_features2]

# Calculate the correlation matrix of the selected features and the "Results" column for the first dataset
corr1 = X_subset1.corr()
# Calculate the correlation matrix of the selected features and the "Results" column for the second dataset
corr2 = X_subset2.corr()

# Print the correlation matrix for the first dataset
print("Correlation matrix for the first dataset:")
print(corr1)
print()

# Print the correlation matrix for the second dataset
print("Correlation matrix for the second dataset:")
print(corr2)
print()

# Create a heatmap of the correlation matrix for the first dataset
sns.heatmap(corr1, cmap='RdYlBu', annot=True)
plt.savefig('heatmap1_RFE_masks.png')
plt.clf()

# Create a heatmap of the correlation matrix for the first dataset
sns.heatmap(corr2, cmap='RdYlBu', annot=True)
plt.savefig('heatmap2 RFE(blankets).png')
plt.clf()