# Assuming that 'em' is the correct import for the EM function in mixem
from mixem import em

# Continue with setting up the data and distributions as before
import numpy as np
from mixem.distribution import NormalDistribution

# Sample Input_LO and Output_LO data (for illustration)
input_LO_data = np.array([
    [1.0, 0.5, -1.0],
    [2.0, 0.3, 0.7],
    [1.5, 1.0, -0.5],
    # ... add more samples
])

output_LO_data = np.array([
    0.75,  # Result of the first fragment measurement
    1.2,   # Result of the second fragment measurement
    0.6,   # Result of the third fragment measurement
    # ... corresponding outputs
])

# Initialize distributions for clustering
distributions = [
    NormalDistribution(np.mean(output_LO_data) - 0.5, 1.0),
    NormalDistribution(np.mean(output_LO_data) + 0.5, 1.0)
]

# Initialize and run the EM algorithm
em_instance = em(output_LO_data, distributions)
print(em_instance)
em_instance.run()

# Print the inferred parameters
for i, dist in enumerate(distributions):
    print(f"Cluster {i}: mean={dist.mean}, variance={dist.variance}")
