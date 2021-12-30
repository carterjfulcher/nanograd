import numpy as np 

from scuti.utils.distribution import Distribution


def test_classifier():
    import pandas as pd 
    import numpy as np
    from scuti.bayes import BayesClassifier

    df = pd.read_csv('sample_data/500_Person_Gender_Height_Weight_Index.csv')
    train_y = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1).values
    train_x = df.drop('Gender', axis=1).values

    model = BayesClassifier()
    model.fit(train_x, train_y)


    assert(isinstance(model._variances, np.ndarray))
    assert(isinstance(model._means, np.ndarray))

if __name__ == "__main__":
    test_classifier()