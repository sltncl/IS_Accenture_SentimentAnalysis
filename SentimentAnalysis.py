from Vectorizer import vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Import the new dataset from NewDataSet.cvs
df = pd.read_csv('DataSet/New2DataSet.csv')
# Set the features and label
# A feature is an input variable
# A label is the thing we're predicting
X = df['review']
y = df['sentiment']

# I will use a count vectorizer to vectorize the text data in the review column
# and then use three different classification models from scikit-learn models.

'''
# The CountVectorizer uses the bag of word approach that ignores the text structures and only extract information from the word counts. 
# It will transform each document into a vector. 
# The inputs of the vector are the occurrence count of each unique word for this document. 
# When having m documents in the corpus, and there are n unique words from all m documents, the CountVectorizer will transform the text data into a m*n sparse matrix.
'''

print('COUNT VECTORIZER')
print('\nLogistic Regression')
# Defines the variable countVectlr as an object Vectorizer
countVectlr = vectorizer(CountVectorizer, LogisticRegression(max_iter=500),X,y)
countVectlr.trainingModel()
print('\nSupport Vector Machine')
# Defines the variable countVectSvm as an object Vectorizer
countVectSvm = vectorizer(CountVectorizer, svm.SVC(),X,y)
countVectSvm.trainingModel()
print('\nKNearestNeighbor')
# Defines the variable countVectNkk as an object Vectorizer
countVectNkk = vectorizer(CountVectorizer, KNeighborsClassifier(),X,y)
countVectNkk.trainingModel()

# Next, I will use the TF-IDF vectorizer to vectorize the text data in the review column and then use three different classification models from scikit-learn models.
# This vectorizer is known to be a more popular one because it uses the term frequency of the words.

'''
# TFIDF is short for term frequency, inverse document frequency. 
# Besides the word counts in each document, TFIDF also includes the occurrence of this word in other documents
'''

print('TFIDF VECTORIZER')
print('\nLogistic Regression')
# Defines the variable TfidfVectlr as an object Vectorizer
TfidfVectlr = vectorizer(TfidfVectorizer, LogisticRegression(),X,y)
TfidfVectlr.trainingModel()
print('\nSupport Vector Machine')
# Defines the variable TfidfVectSvm as an object Vectorizer
TfidfVectSvm = vectorizer(TfidfVectorizer, svm.SVC(),X,y)
TfidfVectSvm.trainingModel()
print('\nKNearestNeighbor')
# Defines the variable TfidfVectKnn as an object Vectorizer
TfidfVectKnn = vectorizer(TfidfVectorizer, KNeighborsClassifier(),X,y)
TfidfVectKnn.trainingModel()