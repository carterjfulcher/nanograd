
from scuti.utils.Distribution import Distribution


def test_classifier():
    import pandas as pd 
    import numpy as np
    from scuti.bayes import BayesClassifier

    df = pd.read_csv('sample_data/500_Person_Gender_Height_Weight_Index.csv')
    labels = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1).values
    features = df.drop('Gender', axis=1).values

    std, mean = Distribution.normal_distribution(df.drop('Gender', axis=1))
    model = BayesClassifier()
    model.fit(labels=labels, features=features)


if __name__ == "__main__":
    test_classifier()