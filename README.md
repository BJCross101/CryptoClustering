# Cryptocurrency Clustering with K-Means and PCA

## Overview
This project aims to cluster cryptocurrencies using K-Means clustering algorithm and Principal Component Analysis (PCA). The clustering is performed on both the original scaled data and the data reduced to three principal components using PCA. The optimal number of clusters is determined using the elbow method for both datasets.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Usage
1. Clone the repository to your local machine.
2. Ensure all required libraries are installed. You can install them using pip: `pip install -r requirements.txt`.
3. Place your original scaled data in a CSV file named `original_scaled_data.csv`.
4. Run the `main.py` script.
5. Review the results generated in the console and the visualizations displayed.

## Files
- `main.py`: Contains the main script for executing the clustering and PCA analysis.
- `original_scaled_data.csv`: Sample dataset containing original scaled cryptocurrency data.
- `README.md`: This file, providing an overview of the project and instructions for usage.

## Tasks Completed
1. **Finding the Best Value for k**:
   - Implemented the elbow method to find the optimal number of clusters.
   - Visualized the inertia values for different values of k.

2. **Clustering Cryptocurrencies with K-Means**:
   - Clustered cryptocurrencies using K-Means algorithm on the original scaled data.
   - Visualized the clusters in a scatter plot.

3. **Optimizing Clusters with PCA**:
   - Performed PCA on the original scaled data to reduce dimensions.
   - Reviewed the explained variance of the principal components.

4. **Finding the Best Value for k Using PCA Data**:
   - Applied the elbow method on the PCA data to determine the best k.
   - Compared the best k obtained with the PCA data to that obtained with the original data.

5. **Clustering Cryptocurrencies with PCA Data**:
   - Clustered cryptocurrencies using K-Means algorithm on the PCA-reduced data.
   - Visualized the clusters in a scatter plot using principal components.

6. **Determining Feature Weights on Principal Components**:
   - Analyzed the weights of each feature on each principal component.

## Conclusion
This project demonstrates the use of K-Means clustering and PCA for clustering cryptocurrencies. The results provide insights into the grouping of cryptocurrencies based on price change percentages.