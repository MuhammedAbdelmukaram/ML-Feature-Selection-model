import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Ad Group Budget': [25, 50, 25, 25, 25, 25],
    'Cost': [48.5, 135.92, 104.54, 69.45, 165.32, 51.18],
    'Impression': [25023, 47128, 48505, 43954, 57815, 19999],
    'Click': [138, 482, 464, 437, 765, 219],
    'CTR': [0.55, 1.02, 0.96, 0.99, 1.32, 1.1],
    'Conversions': [0, 4, 4, 2, 6, 0],
    'CPA': [0, 33.98, 26.14, 34.73, 27.55, 0],
    'CVR': [0, 0.83, 0.86, 0.46, 0.78, 0],
    'Results': [0, 4, 4, 2, 6, 0],
    'Results Rate': [0, 0.83, 0.86, 0.46, 0.78, 0],
    'Frequency': [1.26, 1.27, 1.38, 1.19, 1.31, 1.19],
    'Video Views at 25%': [6305, 13530, 13271, 13886, 16321, 6717],
    'Video Views at 50%': [3593, 7348, 7307, 7444, 9632, 4079],
    'Video Views at 75%': [2233, 4180, 4310, 3899, 6025, 2683],
    'Video Views at 100%': [1523, 2640, 2928, 2492, 4177, 1914],
    '6-Second Video Views': [2411, 6302, 5852, 6218, 7738, 2830],
    'Total Add Payment Info': [2, 7, 4, 2, 8, 0],
    'Total Initiate Checkout': [10, 41, 27, 38, 56, 2],
    'Add to Cart': [18, 77, 47, 61, 72, 8],
    'Total View Content': [104, 444, 401, 405, 715, 204],
    'Total Page View': [91, 382, 368, 358, 634, 188]
}

df = pd.DataFrame(data)

# Calculate the correlation between 'Results' and other columns
corr = df.corr()['Results']

# Drop the 'Results' column
corr = corr.drop('Results')

# Create a DataFrame from the correlation values
corr_matrix = pd.DataFrame(corr, columns=['Results'])

# Create a heatmap using seaborn
sns.heatmap(corr_matrix, cmap='RdYlBu', annot=True)
plt.savefig('heatmap TEST.png')