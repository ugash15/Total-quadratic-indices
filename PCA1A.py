import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import warnings
from mpl_toolkits import mplot3d
from pylab import rcParams
from scipy import stats
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan, het_white


import warnings
warnings.filterwarnings("ignore")
import warnings
import pandas as pd
from matplotlib import pyplot as plt, rcParams

# Suppress warnings
warnings.filterwarnings("ignore")

# Pandas display settings
pd.set_option('display.expand_frame_repr', False)

# Matplotlib settings
rcParams['figure.figsize'] = (16, 8)

df=pd.read_csv('D:\Python code\pythonProject1\ATOMICNUMBER.csv')
print(df)

df.head()

print ("Total number of rows in dataset = {}".format(df.shape[0]))
print ("Total number of columns in dataset = {}".format(df.shape[1]))


# Select columns 3 to 9 (C to I) for dependent variables (y)
y = df.iloc[:, 2:9]  # iloc uses zero-based indexing; column 3 is index 2

# Select columns 10 to 26 (J to Z) for independent variables (X)
X = df.iloc[:, 9:26]  # column 10 is index 9

# Ensure only numeric columns are used
numeric_df = df.select_dtypes(include=['float', 'int'])

# Check if there are non-numeric columns (debugging step)
non_numeric_cols = df.select_dtypes(exclude=['float', 'int']).columns
if not non_numeric_cols.empty:
    print("Non-numeric columns found:", non_numeric_cols.tolist())

# Verify shapes of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Check if the number of rows in X and y are the same
if X.shape[0] != y.shape[0]:
    raise ValueError(f"Mismatch in number of rows: X has {X.shape[0]} rows and y has {y.shape[0]} rows.")

# Generate a heatmap for the independent variables only
def plot_independent_variable_heatmap(X):
    # Compute the correlation matrix for the independent variables
    correlation_matrix = X.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Heatmap of Independent Variables")
    plt.show()

# Call the function with your independent variables
plot_independent_variable_heatmap(X)

# Define the heatmap function
def plot_heatmap(X, y):
    for target_col in y.columns:
        # Concatenate the independent variables (X) with the current dependent variable
        data = pd.concat([X, y[target_col]], axis=1)

        # Compute the correlation matrix
        correlation_matrix = data.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"Heatmap of {target_col} with Independent Variables")
        plt.show()

# Call the function to generate heatmaps
plot_heatmap(X, y)

# Loop over each column in y (if there are multiple dependent variables)
for target_col in y.columns:
    X_with_constant = sm.add_constant(X)  # Add constant to the independent variables
    model = sm.OLS(y[target_col], X_with_constant)  # Fit the model for the specific target column
    results = model.fit()
    print(f"Results for {target_col}:")
    print(results.summary())
    print("\n")

    # Create the OLS table as an image
    fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size for the image
    ax.axis('tight')
    ax.axis('off')

    # Convert OLS summary to a string
    table_str = results.summary().as_text()

    # Use matplotlib's text functionality to display the table
    ax.text(0.01, 0.95, table_str, fontsize=10, ha='left', va='top', family='monospace')

    # Save the table as an image
    plt.savefig(f"OLS_Regression_{target_col}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to avoid displaying it in the notebook
    print(f"OLS table for {target_col} saved as image.\n")

    # Reshape the necessary columns from X (Independent Variables)
    x_line = X.iloc[:, 0].values.reshape(-1, 1)  # First column of X (e.g., 'X1')
    y_line = X.iloc[:, 1].values.reshape(-1, 1)  # Second column of X (e.g., 'X2')
    z_line = X.iloc[:, 2].values.reshape(-1, 1)  # Third column of X (e.g., 'X3')

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Scatter plot for the 3D data points
    ax.scatter3D(x_line, y_line, z_line, color='gray')

    # Set labels for the axes (optional)
    ax.set_xlabel('X Label')  # Label for the x-axis
    ax.set_ylabel('Y Label')  # Label for the y-axis
    ax.set_zlabel('Z Label')  # Label for the z-axis

    # Show the plot
    plt.show()

def plot_plane_with_points(x, y, z):
    # Stack the x and y columns horizontally, then add a column of ones for the intercept
    X = np.hstack((x, y))
    X = np.hstack((np.ones((x.shape[0], 1)), X))  # Adding the intercept column

    # Perform the regression to solve for theta (plane parameters)
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), z)

    # Determine the size of the plane (based on the maximum values of x, y, z)
    k = int(max(np.max(x), np.max(y), np.max(z)))  # Adjust this depending on your data range

    # Create a grid for plotting the plane
    p1, p2 = np.mgrid[:k, :k]
    P = np.hstack((np.reshape(p1, (k * k, 1)), np.reshape(p2, (k * k, 1))))
    P = np.hstack((np.ones((k * k, 1)), P))  # Add the intercept term

    # Compute the values for the plane based on the regression coefficients
    plane = np.reshape(np.dot(P, theta), (k, k))

    # Plotting
    fig = plt.figure()

    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scatter points
    ax.plot(x[:, 0], y[:, 0], z[:, 0], 'ro')  # 'ro' for red dots

    # Plot the regression plane
    ax.plot_surface(p1, p2, plane, alpha=0.5, rstride=100, cstride=100)  # semi-transparent surface

    # Set axis labels
    ax.set_xlabel('X1 Label')  # Customize based on your data
    ax.set_ylabel('X2 Label')  # Customize based on your data
    ax.set_zlabel('Y Label')  # Customize based on your data

    return plt.show()


# Example usage with your data
x_line = X.iloc[:, 0].values.reshape(-1, 1)  # First column of X (e.g., 'X1')
y_line = X.iloc[:, 1].values.reshape(-1, 1)  # Second column of X (e.g., 'X2')
z_line = y.iloc[:, 0].values.reshape(-1, 1)  # One column from y (e.g., the first target column)

# Call the function with your data
plot_plane_with_points(x_line, y_line, z_line)

X_std = StandardScaler().fit_transform(X)

pca = PCA().fit(X_std)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

# Assuming you already have your data loaded in 'df'

# Select only the numeric columns (adjust if needed)
numeric_data = df.select_dtypes(include=['float', 'int'])

# Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Perform PCA (adjust the number of components as needed)
pca = PCA(n_components=len(numeric_data.columns))
pca.fit(scaled_data)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Compute the cumulative sum of the explained variance ratio
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Print the cumulative explained variance
print(cumulative_explained_variance)

# Standardize the independent variables
scaler = StandardScaler()
X_std = scaler.fit_transform(X)  # X is your independent variables

# Perform PCA with 1 component
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)  # Transform the data to 1 component

# Output the transformed data (first 5 rows)
print("Transformed data (first 5 rows):")
print(X_pca[:5])  # You can adjust this to see more or less data

# Loop over each column in 'y' (since it has multiple dependent variables)
for target_col in y.columns:
    # Add constant to the PCA-transformed data
    X_pca_with_constant = sm.add_constant(X_pca)

    # Fit the OLS model for the current target column
    model = sm.OLS(y[target_col], X_pca_with_constant)

    # Fit the model and print the summary
    results = model.fit()
    print(f"Results for {target_col}:")
    print(results.summary())
    print("\n")

    # Create the OLS table as an image
    fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size for the image
    ax.axis('tight')
    ax.axis('off')

    # Convert OLS summary to a string
    table_str = results.summary().as_text()

    # Use matplotlib's text functionality to display the table
    ax.text(0.01, 0.95, table_str, fontsize=10, ha='left', va='top', family='monospace')

    # Save the table as an image
    plt.savefig(f"OLS_Regression_{target_col}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to avoid displaying it in the notebook
    print(f"OLS table for {target_col} saved as image.\n")

    # Extract PCA loadings
    loadings = pd.DataFrame(pca.components_, columns=X.columns,
                            index=[f'PC{i + 1}' for i in range(len(pca.components_))])

    # Display the loadings
    print("PCA Loadings:")
    print(loadings)

    # Identify the dominant variables for PC1
    pc1_loadings = loadings.loc['PC1']
    dominant_variables = pc1_loadings[pc1_loadings.abs() > 0.25].index  # Try with a lower threshold
    print(f"The dominant variables contributing to PC1 (x1) are: {list(dominant_variables)}")

    print("Explained variance ratio for each PC:")
    print(pca.explained_variance_ratio_)


# Loop over each column in 'y' (multiple dependent variables)
for target_col in y.columns:
    # Fit the model for the current target column (like before)
    X_pca_with_constant = sm.add_constant(X_pca)
    model = sm.OLS(y[target_col], X_pca_with_constant)
    results = model.fit()

    # Plot the scatter plot and the regression line for the current target column
    plt.scatter(X_pca, y[target_col], label='Data', color='red')  # scatter plot
    plt.plot(X_pca, results.predict(X_pca_with_constant), color='blue', label='Regression Line')  # regression line
    plt.xlabel('PCA Component 1')  # X-axis label
    plt.ylabel(target_col)  # Y-axis label (dependent variable)
    plt.title(f"Scatter Plot and Regression Line for {target_col}")
    plt.legend()
    plt.show()



