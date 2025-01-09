# Kernel_SVM_With_Python

## Dataset
_The First 15 rows of about 400 rows of data is displayed in the table_

|Age|	Estimated Salary|	Purchased|
|----|-----------------|---------|
|19|	19000|	0|
|35	|20000	|0|
|26|	43000|	0|
|27	|57000	|0|
|19|	76000|	0|
|27	|58000	|0|
|27|	84000|	0|
|32	|150000	|1|
|25|	33000|	0|
|35	|65000	|0|
|26|	80000|	0|
|26	|52000	|0|
|20|	86000|	0|
|32	|18000	|0|

## About the Dataset
_The dataset is used to predict a categorical variable of Yes and No where the categories are encoded to be 1 (for Yes) and 0 (for No). This is to use the variable Age and Estimated Salary as the predictors. It contains 400 observations_

## Case Scenario
Your company where you work as a data scientist just released a new brand of furniture, and your role is to predict which of your customers can purchased the product. The result will help the advertising team target the right customers that will buy the new product.

## Model used
This model uses the Support Vector Machine (SVM) to carryout the prediction.

## Python codes for Support Vector Machine (SVM) Model

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
```

### Predicting a new result

```python
print(classifier.predict(sc.transform([[30,87000]])))
```

### Predicting the Test set results
```python
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

### Making the Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

+ [See code in terminal, click to view](https://colab.research.google.com/drive/1LypxD9fmsHC-fSUN2xr6j120oElbCbQQ#scrollTo=e0pFVAmciHQs)

## Quesions on SVM
[Click to view Question](https://ibb.co/bJGzRVd)

### Answers
+ [See Answer One](https://ibb.co/Qvh9m2q)
+ [See Answer Two](https://ibb.co/bHCZW6H)
+ [See Answer Three](https://ibb.co/hB0xfLq)
+ [See Answer Four](https://ibb.co/nc4dgYS)
+ [See Answer Five](https://ibb.co/X8L1Lq9)
