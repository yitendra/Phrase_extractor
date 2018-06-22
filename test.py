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

#Setting mycopus.txt
mycorpus=['i','me','at','to','date','time','for','tommorow','tonight','today',\
           'sunday','monday','tuesday','wednesday','thursday','friday','saturday',\
           'morning','evening']
#For found, extract result text
def extract(index):
    #Create the extractor function
    temp=X[index].lower()
    t=re.findall('remi[a-z]+ me? to? (.+?) at',temp)
    if len(t)==0:
        t=re.findall('rem[a-z]+ me? to? (.+?) on',temp)
    if len(t)==0:
        t=temp
        t+=' .y'
        t=re.findall('remi[\w.]+ (.+?) {}'.format(t.split(" ")[len(t.split(" "))-1]),t)
    if len(t)>0:
        t=str(t[0]).split()    
        t=[word for word in t if not word in mycorpus]
    if len(t)>0:
        result[index]=" ".join(t)
    else:
        result[index]="Not Found"

for i in range(len(X)):
    if(result[i]=="Found"):
        extract(i)
        
#----Saving result------------
file=open("Result.tsv",'w')
file.write("sent\tlabel\n")
for i in range(len(result)):
    file.write(str(X[i])+'\t'+str(result[i])+'\n')
file.close()
#---------------------------------------