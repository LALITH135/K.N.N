

# Implementing a KNN model to classify the animals in to categorie


import pandas as pd
# import dataset
df = pd.read_csv('C:\python notes\ASSIGNMENTS\K.N.N\Zoo.csv')

df.shape
df.head()
list(df)
df.info()

# split as X and Y
Y = df['type']
X = df.iloc[:,:17]
list(X)

# label encoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['animal name'] = LE.fit_transform(df['animal name'])
df['animal name'].value_counts()


# standardization
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)

pd.crosstab(Y,Y)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y,stratify=Y ,random_state=42)  # By default test_size=0.25

pd.crosstab(Y,Y)
pd.crosstab(y_train,y_train)


# Install KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, y_train)

# Prediction
y_pred=knn.predict(X_test)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy score of knn with k=5

accuracy_score(y_test, y_pred)
print('Accuracy of KNN with K=5, on the test set: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# to check with other k values
# forloop

k_num = range(1,15,1)

Test_accuracy=[]
for i in k_num:
    knn = KNeighborsClassifier(n_neighbors=i,p=2)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    Test_accuracy.append(accuracy_score(y_test, y_pred).round(3))

print(Test_accuracy)
Test_accuracy

'''
print(Test_accuracy)
[1.0, 0.962, 0.962, 0.962, 0.923, 0.923, 0.885, 0.885, 0.846, 0.885, 0.885, 0.923, 0.923, 0.923]
'''

# for knn plot
import matplotlib.pyplot as plt
plt.plot(k_num,Test_accuracy,)
plt.show()

#---------------------------------------------------------------------















