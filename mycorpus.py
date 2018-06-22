from nltk.corpus import stopwords
import pandas as pd
import re
import pickle

#getting english stopwords
mycorpus=list()
#for words in stopwords.words('english'):
 #   mycorpus.append(words)

#Analys which word is not in label
#getting data
df=pd.read_csv("training_data.tsv",delimiter='\t')
df=df.dropna()
y=df['label']
X=df['sent']

X_list=list()
for _ in X:
    t=re.findall('[a-z]+',_.lower())
    t=[word for word in t]
    X_list+=t
X_list=set(X_list)

y_list=list()
for _ in y:
    t=re.findall('[a-z]+',_.lower())
    t=[word for word in t]
    y_list+=t
y_list=set(y_list)

#Saving mycorpus in pickle
mycorpus+=list(X_list-y_list)
cfile=open("mycorpus.pickle",'wb')
pickle.dump(mycorpus,cfile)
cfile.close()

#Saving mycorpus in file
file = open('mycorpus.txt','w')
for _ in mycorpus:
    file.write(_+'\n')
file.close()