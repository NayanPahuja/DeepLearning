import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and prepare the data
def load_and_prepare_data():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['PRICE'] = california.target
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df.drop('PRICE', axis=1))
    df_scaled = pd.DataFrame(features_scaled, columns=df.drop('PRICE', axis=1).columns)
    df_scaled['PRICE'] = df['PRICE']
    
    return df_scaled

# Function to calculate regression statistics
def calculate_regression_stats(X, y):
    model = LinearRegression()
    model.fit(X, y)
    n = X.shape[0]
    p = X.shape[1]
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (n - p - 1)
    
    X_with_const = np.column_stack([np.ones(n), X])
    var_covar_matrix = mse * np.linalg.inv(X_with_const.T.dot(X_with_const))
    
    se = np.sqrt(np.diag(var_covar_matrix))[1:]
    t_scores = model.coef_ / se
    p_values = 2 * (1 - stats.t.cdf(abs(t_scores), df=n - p - 1))
    
    r_squared = model.score(X, y)
    
    stats_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Std_Error': se,
        't_score': t_scores,
        'p_value': p_values
    })
    
    return stats_df, r_squared

# Function to plot statistical significance (T-scores and P-values)
def plot_statistical_significance(stats_df):
    plt.figure(figsize=(12, 6))
    
    # Plot T-scores
    plt.subplot(1, 2, 1)
    sns.barplot(x='Feature', y='t_score', data=stats_df)
    plt.xticks(rotation=45)
    plt.title('T-scores by Feature')
    
    # Plot P-values
    plt.subplot(1, 2, 2)
    sns.barplot(x='Feature', y='p_value', data=stats_df)
    plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05 threshold')
    plt.xticks(rotation=45)
    plt.title('P-values by Feature')
    
    plt.tight_layout()
    plt.show()

# Function for feature selection based on statistical significance (p < 0.05)
def feature_selection(stats_df, significance_level=0.05):
    significant_features = stats_df[stats_df['p_value'] < significance_level]
    return significant_features

# Main function to load data, train the model, and display results
def main():
    # Load and prepare the data
    df = load_and_prepare_data()
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate regression statistics
    stats_df, r_squared = calculate_regression_stats(X_train, y_train)
    print("\nRegression Statistics:")
    print("=====================")
    print(f"\nR-squared: {r_squared:.4f}")
    print("\nFeature Statistics:")
    print(stats_df.round(4))
    
    # Feature selection based on p-value < 0.05
    significant_features = feature_selection(stats_df)
    print("\nSignificant Features (p < 0.05):")
    print("==============================")
    print(significant_features[['Feature', 'Coefficient', 'p_value']].round(4))
    
    # Plot statistical significance
    plot_statistical_significance(stats_df)
    
    # Reduce features to only significant ones
    X_reduced = X_train[significant_features['Feature']]
    
    # Calculate regression statistics for reduced model
    reduced_stats, reduced_r2 = calculate_regression_stats(X_reduced, y_train)
    print("\nModel Performance Comparison:")
    print("===========================")
    print(f"Full model R-squared: {r_squared:.4f}")
    print(f"Reduced model R-squared: {reduced_r2:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
