from pyem import EM
import numpy as np

# Example data
output_LO_data = np.array([0.75, 1.2, 0.6, 0.9, 1.1, 0.5, 0.8, 1.3])

# Reshape data for EM fitting
output_LO_data = output_LO_data.reshape(-1, 1)

# Initialize the EM object with desired number of components
em_model = EM(output_LO_data, num_components=2)

# Fit the model to the data
em_model.fit()

# Output the learned parameters (e.g., means, variances)
print("Means:", em_model.means)
print("Covariances:", em_model.covariances)
