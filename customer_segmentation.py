import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)

# Create Cluster column
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Show first few rows
print(df.head())

# Save new file
df.to_csv("customer_segments.csv", index=False)

print("\nCustomer segments created successfully!")
print(df.columns)
print(df['Cluster'].value_counts())