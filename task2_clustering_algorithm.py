import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Choose features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Convert cluster centers back to original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Map clusters to descriptive labels
cluster_labels = {
    0: 'High Income, Low Spending',
    1: 'Low Income, Low Spending',
    2: 'Low Income, High Spending',
    3: 'Average Income & Spending',
    4: 'High Income, High Spending'
}
data['Customer_Type'] = data['Cluster'].map(cluster_labels)

# Show a sample
print("\n✅ Sample Customers with Labels:\n")
print(data[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', 'Customer_Type']].head())

# Plot clusters using Matplotlib
plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green', 'orange', 'purple']

for cluster_id in range(5):
    cluster_points = data[data['Cluster'] == cluster_id]
    plt.scatter(
        cluster_points['Annual Income (k$)'],
        cluster_points['Spending Score (1-100)'],
        s=70,
        c=colors[cluster_id],
        label=cluster_labels[cluster_id],
        alpha=0.6
    )

# Plot centroids
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Mall Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
