import numpy as np
from sklearn.mixture import GaussianMixture

# Sample data for Output_LO (the observed data)
output_LO_data = np.array([0.75, 1.2, 0.6, 0.9, 1.1, 0.5, 0.8, 1.3])

# Reshape the data to fit the GMM model (required by Scikit-learn)
output_LO_data = output_LO_data.reshape(-1, 1)

# Fit a Gaussian Mixture Model (GMM) to the data using EM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(output_LO_data)

# Print the learned means and covariances
print(f"Means: {gmm.means_}")
print(f"Covariances: {gmm.covariances_}")

# Predict the probabilities of each data point belonging to each component
probs = gmm.predict_proba(output_LO_data)
print("Probabilities of each point belonging to each cluster:")
print(probs)