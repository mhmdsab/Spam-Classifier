import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from string import punctuation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack,csr_matrix
from sklearn.svm import SVC




#import data
data = pd.read_pickle('spam.pickle')
data = data[['v2','v1']]

#drop nan values
data.dropna(inplace = True)

#creating new feature which is number of digits in each e-mail
data['cnt_dig'] = data['v2'].str.count(r'\d') #number of digits
data['cnt_non_word'] = data['v2'].str.count(r'[^A-Za-z_\d]') #number of non-word characters (anything other than a letter, digit or underscore)
num_dig = data['cnt_dig'].values
num_non_word = data['cnt_non_word'].values

#changing numbers to num word
data['v2'] = data['v2'].str.replace(r'\d',' number ')

#labelizing ratings
le = LabelEncoder()
data['v1'] = le.fit_transform(data['v1'])

#removing stop words & stemming words
review_array = np.array(data['v2'])
corpus = []
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

for i in range(len(data)):
    review = review_array[i]
    review = word_tokenize(review)
    review = [w for w in review if not w in stop_words]
    review = [ps.stem(w) for w in review]
    review = ' '.join(review)
    corpus.append(review)
   

#creating feature matrices
cv = TfidfVectorizer(min_df = 5,ngram_range = (1,5),max_features=2000)
X = cv.fit_transform(corpus).toarray()
X = hstack([X, csr_matrix(num_dig).T], 'csr')
X = hstack([X, csr_matrix(num_non_word).T], 'csr')

#or u can use numpy instead of scipy

#X = np.hstack((X, num_dig.reshape(-1,1)))
#X = np.hstack((X, num_non_word.reshape(-1,1)))

y = data['v1'].values

feature_names = cv.get_feature_names()
print('number of features observed', len(feature_names))

#cross_validation
X_train,X_cv,y_train,y_cv = train_test_split(X,y,test_size = 0.2)

#fit the model
#clf = MultinomialNB(alpha = 0.01)
clf = SVC(C=1000)
clf.fit(X_train,y_train)

#prediction
y_pred = clf.predict(X_cv)

#score
print(roc_auc_score(y_cv,y_pred))
print(clf.score(X_cv,y_cv))


#confusion matrix
cm = confusion_matrix(y_cv,y_pred)
print(cm)

pickling_on = open("clf.pickle","wb")
pickle.dump(clf, pickling_on)
pickling_on.close()
