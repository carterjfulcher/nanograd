import numpy as np 

class Distribution: 
    @classmethod 
    def normal_distribution(self, x: np.ndarray) -> tuple: 
        means, stds = [], []
        for feature_set_index in range(x.shape[1]):
            feature_set = [i[feature_set_index] for i in x]
            means.append(np.mean(feature_set))
            stds.append(np.std(feature_set))

        return (sum(means) / len(means)), (sum(stds) / len(stds)) #σ = standard deviation, μ = mean