# Scuti

A filtering and prediction library

## How to use

```python
from scuti.classifiers import BayesClassifier

# load in some sample data and do some processing
df = pd.read_csv('sample_data/500_Person_Gender_Height_Weight_Index.csv')
train_y = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1).values
train_x = df.drop('Gender', axis=1).values

#create and train the model
model = BayesClassifier()
model.fit(train_x, train_y)

#inference
model.predict([143, 59, 4]) # --> 0

```
