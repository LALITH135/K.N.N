

# model for glass classification using KNN

import pandas as pd
# import dataset
df = pd.read_csv('C:\python notes\ASSIGNMENTS\K.N.N\glass.csv')

df.shape
list(df)
df.head()
df.info()

# split as X and Y
Y = df['Type']
X = df.iloc[:,:9]
list(X)

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

'''
print(Test_accuracy)
[0.741, 0.796, 0.778, 0.722, 0.741, 0.667, 0.704, 0.722, 0.741, 0.704, 0.685, 0.648, 0.648, 0.667]
'''
# for knn plot
import matplotlib.pyplot as plt
plt.plot(k_num,Test_accuracy,)
plt.show()

#--------------------------------------------------------------------





























