# %%
# Import required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load the data into a Pandas DataFrame and make the index the "coin_id" column.
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")

# Display sample data
market_data_df.head(10)

# %%
# Generate summary statistics
market_data_df.describe()

# %% [markdown]
# ### Prepare the Data

# %%
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled = StandardScaler().fit_transform(market_data_df[["price_change_percentage_24h", "price_change_percentage_7d", 
                                                                    "price_change_percentage_14d", "price_change_percentage_30d", 
                                                                    "price_change_percentage_60d", "price_change_percentage_200d",
                                                                    "price_change_percentage_1y"]])

# %%
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(market_data_scaled,
                                    columns=["price_change_percentage_24h", "price_change_percentage_7d", 
                                            "price_change_percentage_14d", "price_change_percentage_30d", 
                                            "price_change_percentage_60d", "price_change_percentage_200d",
                                            "price_change_percentage_1y"])
# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = market_data_df.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled

# %% [markdown]
# ### Find the Best Value for k Using the Original Scaled DataFrame.

# %%
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using the scaled DataFrame
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, n_init='auto', random_state=1)
    k_model.fit(market_data_df)
    inertia.append(k_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

# Display the DataFrame
df_elbow


# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.plot.line(x="k",
                   y="inertia",
                   title="Elbow Curve",
                   xticks=k)


# %% [markdown]
# #### Answer the following question: 
# **Question:** What is the best value for `k`?
# 
# **Answer:** 3

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the Original Scaled Data.

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=3, n_init='auto', random_state=1)

# %%
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)

# %%
# Predict the clusters to group the cryptocurrencies using the scaled data
kmeans_predictions = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
kmeans_predictions


# %%
# Create a copy of the DataFrame
df_market_data_kmeans_predictions = df_market_data_scaled.copy()

# %%
# Add a new column to the DataFrame with the predicted clusters
df_market_data_kmeans_predictions["kmeans_predictions"] = kmeans_predictions

# Display sample data
df_market_data_kmeans_predictions.head()


# %%
# Create a scatter plot using Pandas plot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`.
# Use "rainbow" for the color to better visualize the data.
df_market_data_kmeans_predictions.plot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    c="kmeans_predictions",
    colormap="rainbow")

# %% [markdown]
# ### Optimize Clusters with Principal Component Analysis.

# %%
# Create a PCA model instance and set `n_components=3`.
pca=PCA(n_components=3)


# %%
# Use the PCA model with `fit_transform` on the original scaled DataFrame to reduce to three principal components.
market_data_pca = pca.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
market_data_pca[:5]

# %%
# Retrieve the explained variance to determine how much information  can be attributed to each principal component.
pca.explained_variance_ratio_

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** The total explained variance of the three principal components is approximately 0.895.

# %%
# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data
df_crypto_pca = pd.DataFrame(market_data_pca, columns=["PCA1", "PCA2", "PCA3"])

# Copy the crypto names from the original data
df_crypto_pca["coin_id"] = market_data_df.index

# Set the coinid column as index
df_crypto_pca = df_crypto_pca.set_index("coin_id")

# Display sample data
df_crypto_pca


# %% [markdown]
# ### Find the Best Value for k Using the PCA Data

# %%
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using PCA DataFrame.
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, n_init='auto', random_state=2)
    k_model.fit(df_crypto_pca)
    inertia.append(k_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

# Display the DataFrame
df_elbow



# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.plot.line(x="k",
                   y="inertia",
                   title="Elbow Curve",
                   xticks=k)

# %% [markdown]
# #### Answer the following questions: 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** Yes, it was originally 3

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=3, n_init='auto', random_state=1)

# %%
# Fit the K-Means model using the PCA data
model.fit(df_market_data_scaled)

# %%
# Predict the clusters to group the cryptocurrencies using the scaled data
kmeans_predictions = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
kmeans_predictions


# %%
# Creating a DataFrame with the PCA data
df_crypto_pca = pd.DataFrame(market_data_pca, columns=["PCA1", "PCA2", "PCA3"])

# Copy the crypto names from the original data
df_crypto_pca["coin_id"] = market_data_df.index

# Set the coinid column as index
df_crypto_pca = df_crypto_pca.set_index("coin_id")

# Display sample data
df_crypto_pca

# %%
# Create a copy of the DataFrame with the PCA data
df_crypto_pca_copy = df_crypto_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_crypto_pca_copy["kmeans_predictions"] = kmeans_predictions

# Display sample data
df_crypto_pca_copy

# %%
# Create a scatter plot using hvPlot by setting `x="PCA1"` and `y="PCA2"`. 
df_crypto_pca_copy.plot.scatter(
    x="PCA1",
    y="PCA2",
    c="kmeans_predictions",
    colormap="winter")

# %% [markdown]
# ### Determine the Weights of Each Feature on each Principal Component

# %%
# Use the columns from the original scaled DataFrame as the index.
pca_component_weights = pd.DataFrame(pca.components_.T, columns=['PCA1', 'PCA2', 'PCA3'], index=df_market_data_scaled.columns)
pca_component_weights

# %% [markdown]
# #### Answer the following question: 
# 
# * **Question:** Which features have the strongest positive or negative influence on each component? 
#  
# * **Answer:** 
# 
# For PCA1, the features "price_change_percentage_200d" and "price_change_percentage_1y" have the strongest positive influence, while "price_change_percentage_24h" has the strongest negative influence.
# 
# For PCA2, "price_change_percentage_30d" and "price_change_percentage_14d" have the strongest positive influence, while "price_change_percentage_1y" has the strongest negative influence.
# 
# For PCA3, "price_change_percentage_7d" has the strongest positive influence, while "price_change_percentage_7d" and "price_change_percentage_60d" have the strongest negative influence.
#     

# %%



