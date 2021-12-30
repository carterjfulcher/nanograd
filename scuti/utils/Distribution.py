import pandas as pd 

class Distribution: 
    @classmethod 
    def normal_distribution(self, df: pd.DataFrame) -> tuple: 
        means, stds = [], []
        for column in df.columns:
            means.append(df[column].mean())
            stds.append(df[column].std())

        return (sum(means) / len(means)), (sum(stds) / len(stds) ** 2) #σ = standard deviation, μ = mean