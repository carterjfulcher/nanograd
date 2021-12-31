import numpy as np 

def test_classifier():
    import pandas as pd 
    import numpy as np
    from scuti.classifiers import BayesClassifier

    df = pd.read_csv('sample_data/500_Person_Gender_Height_Weight_Index.csv')
    train_y = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1).values
    train_x = df.drop('Gender', axis=1).values

    model = BayesClassifier()
    model.fit(train_x, train_y)
    
    assert(isinstance(model._variances, np.ndarray))
    assert(isinstance(model._means, np.ndarray))

    test_x = np.array([187, 89, 2])

    test_y = model.forward(test_x)

    assert(test_y in [0, 1])

if __name__ == "__main__":
    test_classifier()