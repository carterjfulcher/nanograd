from scuti.classifiers import BayesClassifier 
import pandas as pd 

TRAIN_TEST_SPLIT = .90

df = pd.read_csv('sample_data/500_Person_Gender_Height_Weight_Index.csv')
y_set = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1).values
x_set = df.drop('Gender', axis=1).values

split_index = int(TRAIN_TEST_SPLIT * y_set.shape[0])
print(split_index)

x_train = x_set[split_index:] 
y_train = y_set[split_index:] 

x_test= x_set[:split_index]
y_test = y_set[:split_index]


model = BayesClassifier() 

model.fit(x_train, y_train) 

correct = 0

for x, y in zip(x_test, y_test):
  if model.forward(x) == y: correct+=1

print(f'accuracy: {correct / x_test.shape[0]}')
