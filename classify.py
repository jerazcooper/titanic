import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_accuracy(clf, data):
    count = 0
    total = 0
    while total < 10000:
        data['is_train'] = np.random.uniform(0, 1, len(data)) <= .8

        train, test = data[data['is_train'] == True], data[data['is_train'] == False]
        features = data.columns[0:3]
        clf.fit(train[features], train['Survived'])

        for i, j in zip(clf.predict(test[features]), test['Survived']):
            total += 1
            if i == j:
                count += 1
    return count / total

df = pd.read_csv('train.csv')

data = pd.concat([df['Sex'], df['Pclass'], df['Age'], df['Survived']], axis=1)

data['Age'] = data['Age'].fillna(np.mean(data['Age']))
data['Sex'] = [1 if x == 'male' else 0 for x in data['Sex']]

clf = RandomForestClassifier(n_jobs=2)
print(test_accuracy(clf, data))
