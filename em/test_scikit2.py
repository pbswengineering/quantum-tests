import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Create hard-coded data for Input_HI and Input_LO that are correlated
# We'll generate two correlated 2D distributions

np.random.seed(42)  # Set a random seed for reproducibility

# Generate synthetic data where Input_HI and Input_LO are correlated
# Input_HI: Whole circuit data (e.g., classical results after quantum computation)
# Input_LO: Circuit fragment data (e.g., partial results from quantum processors)

# Generate 100 samples with some correlation between Input_HI and Input_LO
x = np.random.randn(100)  # 100 random samples from normal distribution for HI and LO
y = 2 * x + 0.5 * np.random.randn(100)  # LO is linearly correlated with HI plus some noise

# Stack the two data sets into the full data for the EM model
input_HI = np.column_stack([x, np.random.randn(100)])  # Add randomness to make the second feature of HI
input_LO = np.column_stack([y, np.random.randn(100)])  # Add randomness to make the second feature of LO

# Combine the HI and LO inputs into one dataset for the EM algorithm
data = np.hstack([input_HI, input_LO])

# Initialize the GaussianMixture model
n_components = 2  # Number of Gaussian components (adjustable based on your data)
gmm = GaussianMixture(n_components=n_components, random_state=42)

# Fit the model to the data
gmm.fit(data)

# Get the predicted labels (clusters) and the means of the Gaussian components
labels = gmm.predict(data)
means = gmm.means_

# Print the means of the Gaussian components
print(f"Means of the Gaussian components: {means}")

# Model evaluation (AIC and BIC to assess fit quality)
aic = gmm.aic(data)
bic = gmm.bic(data)

# Print AIC and BIC scores
print(f"AIC: {aic}")
print(f"BIC: {bic}")

# IF statement to evaluate the fit quality based on AIC and BIC
if aic < 200:  # Threshold for a good fit (you can adjust the threshold based on your data)
    print("The Gaussian Mixture Model fits the data well based on AIC.")
else:
    print("The Gaussian Mixture Model does not fit the data well based on AIC.")

if bic < 200:  # Threshold for a good fit (you can adjust the threshold based on your data)
    print("The Gaussian Mixture Model fits the data well based on BIC.")
else:
    print("The Gaussian Mixture Model does not fit the data well based on BIC.")

# Visualization (Optional)
# Plot the data and the Gaussian components (for visualization purposes)
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6, label='Data points')
plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Gaussian means')

# Set plot labels and title
plt.xlabel('Input Feature 1')
plt.ylabel('Input Feature 2')
plt.title('EM Algorithm - Correlated Data (Gaussian Mixture)')
plt.legend()
plt.show()

# IF statement to assess if the two distributions are correlated
# Checking if the AIC and BIC indicate a good fit, meaning the two distributions might be correlated
if aic < 200 and bic < 200:
    print("The two distributions are likely correlated.")
else:
    print("The two distributions are likely not correlated.")
