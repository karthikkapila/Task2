import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

customers = 100

data = {
    'Customer_ID': range(1, customers + 1),
    'Annual_Income': np.random.randint(20000, 100000, customers),
    'Purchase_History': np.random.randint(1, 50, customers),
    'Spending_Score': np.random.randint(1, 100, customers)
}

df = pd.DataFrame(data)

print("Customer Dataset:")
print(df.head())

X = df[['Annual_Income', 'Purchase_History', 'Spending_Score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCustomer Groups:")
print(df.head(10))

plt.figure(figsize=(8,6))

plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster'],
    cmap='rainbow',
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Retail Customer Segmentation using K-Means")

plt.show()