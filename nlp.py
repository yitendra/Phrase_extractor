import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
import re
import pickle
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

#getting data
df=pd.read_csv("training_data.tsv",delimiter='\t')
df=df.dropna()
y=df['label']
X=df['sent']

#Processing Data of X
x=list()
for _ in X:
    t=re.findall('[a-z]+',_.lower())
    t=[word for word in t if not word in set(stopwords.words('english'))]
    x.append(" ".join(t))
x=pd.Series(x)

#processing data of y
Y=list()
for _ in y:
    if _ =="Not Found":
        Y.append("Not Found")
    else:
        Y.append("Found")
y=pd.Series(Y)  

#splitting data for train and test
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#Making list of x
x_train_list=list()
for i in X_train:
    x_train_list.append(i)
x_test_list=list()
for j in X_test:
    x_test_list.append(j)

#Vectorizing x
vectorizer = CountVectorizer(max_features=1000) # n-grams Bag of word
train_data = vectorizer.fit_transform(x_train_list) # expects a list of strings
np.asarray(train_data)
test_data = vectorizer.fit_transform(x_test_list) # expects a list of strings
np.asarray(test_data)

#Modelling
#split=train_data,test_data,y_train,y_test
clf=svm.LinearSVC()
clf=clf.fit(train_data,y_train)

#Saving Model in pickle ("model.pickle")
f=open("model.pickle",'wb')
pickle.dump(clf, f)
f.close()

#Checking Accuracy
accuracy=clf.score(test_data,y_test)

#Predicting result
result=clf.predict(test_data)
result_list=list(result)

#creating confusion matrix
cm=confusion_matrix(y_test,result)

# saving file of x_test_list,y_test_list,result_list
y_test_list=list(y_test)
file=open("y_test_result.tsv",'w')
file.write("sent\tlabel\ttest\n")
for i in range(len(y_test)):
    file.write(x_test_list[i]+'\t'+y_test_list[i]+'\t'+result_list[i]+'\n')