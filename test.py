import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

#Loading trained model
f=open("model.pickle",'rb')
clf=pickle.load(f)

#Loading eval file
eval_data=open("eval_data.txt",'r')
X=list()
all_words=list()
for lines in eval_data:
    all_words+=(lines.strip('\n').split(' '))
    X.append(lines.strip('\n'))
eval_data.close()
all_words=list(set(all_words))

#Vectorizing and converting to array: X_test
vectorizer=CountVectorizer(max_features=1000)
X_test=vectorizer.fit_transform(X).toarray()

#Predicting Result (found/not found)
result=list(clf.predict(X_test))

#----Saving temporary result------------
file=open("pre_res.tsv",'w')
file.write("sent\tlabel\n")
for i in range(len(result)):
    file.write(X[i]+'\t'+result[i]+'\n')
file.close()

#getting english stopwords
from nltk.corpus import stopwords
mycorpus=list()
for words in stopwords.words('english'):
    mycorpus.append(words)

#For found, extract result text
def extract(index):
    #Create the extractor function
    pass

for i in range(len(result)):
    if(result[i]=="Found"):
        #extract(i)
        pass
    
