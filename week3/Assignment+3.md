
---

_You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._

---

# Assignment 3

In this assignment you will explore text message data and create models to predict if a message is spam or not. 


```python
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ok lar... Joking wif u oni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FreeMsg Hey there darling it's been 3 week's n...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Even my brother is not like to speak with me. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>As per your request 'Melle Melle (Oru Minnamin...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WINNER!! As a valued network customer you have...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Had your mobile 11 months or more? U R entitle...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
spam_data.count()
```




    text      5572
    target    5572
    dtype: int64




```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
```

### Question 1
What percentage of the documents in `spam_data` are spam?

*This function should return a float, the percent value (i.e. $ratio * 100$).*


```python
def answer_one():
    count = 0
    for i in spam_data['target'] :
        if i == 1 :
            count=count+1
    spam = count/len(spam_data['target'])
    return spam #Your answer here
```


```python
answer_one()
```




    0.13406317300789664



### Question 2

Fit the training data `X_train` using a Count Vectorizer with default parameters.

What is the longest token in the vocabulary?

*This function should return a string.*


```python
from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    # List all tokens and their counts as a dictionary:
    vocabulary = CountVectorizer().fit(X_train).vocabulary_
    
    # You want only the keys, i.e, the words:
    vocabulary = [x for x in vocabulary.keys()]
    
    # Store the lengths in a separate list:
    len_vocabulary = [len(x) for x in vocabulary]
    
    # Use the index of the longest token:
    return vocabulary[np.argmax(len_vocabulary)]
```


```python
answer_two()
```




    'com1win150ppmx3age16subscription'



### Question 3

Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.

Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    cv = CountVectorizer().fit(X_train)
    
    # Transform both X_train and X_test with the same CV object:
    X_train_cv = cv.transform(X_train)
    X_test_cv = cv.transform(X_test)
    
    # Classifier for prediction:
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_cv, y_train)
    preds_test = clf.predict(X_test_cv)
    
    return roc_auc_score(y_test, preds_test)
```


```python
answer_three()
```




    0.97208121827411165



### Question 4

Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.

What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?

Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.

The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 

*This function should return a tuple of two series
`(smallest tf-idfs series, largest tf-idfs series)`.*


```python
from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    #handle tfidf
    tfidf = TfidfVectorizer().fit(X_train)
    #take feature_names out and make it a np array
    feature_names = np.array(tfidf.get_feature_names())
    #transform X_train with rfidf
    X_train_tf = tfidf.transform(X_train)
    #now lets take maxes as row array
    max_tf_idfs = X_train_tf.max(0).toarray()[0] # Get largest tfidf values across all documents.
    sorted_tf_idxs = max_tf_idfs.argsort() # Sorted indices
    sorted_tf_idfs = max_tf_idfs[sorted_tf_idxs] # Sorted TFIDF values
    
    # feature_names doesn't need to be sorted! You just access it with a list of sorted indices!
    smallest_tf_idfs = pd.Series(sorted_tf_idfs[:20], index=feature_names[sorted_tf_idxs[:20]])                    
    largest_tf_idfs = pd.Series(sorted_tf_idfs[-20:][::-1], index=feature_names[sorted_tf_idxs[-20:][::-1]])
    
    return (smallest_tf_idfs, largest_tf_idfs)
```


```python
answer_four()
```




    (sympathetic     0.074475
     healer          0.074475
     aaniye          0.074475
     dependable      0.074475
     companion       0.074475
     listener        0.074475
     athletic        0.074475
     exterminator    0.074475
     psychiatrist    0.074475
     pest            0.074475
     determined      0.074475
     chef            0.074475
     courageous      0.074475
     stylist         0.074475
     psychologist    0.074475
     organizer       0.074475
     pudunga         0.074475
     venaam          0.074475
     diwali          0.091250
     mornings        0.091250
     dtype: float64, 146tf150p    1.000000
     havent       1.000000
     home         1.000000
     okie         1.000000
     thanx        1.000000
     er           1.000000
     anything     1.000000
     lei          1.000000
     nite         1.000000
     yup          1.000000
     thank        1.000000
     ok           1.000000
     where        1.000000
     beerage      1.000000
     anytime      1.000000
     too          1.000000
     done         1.000000
     645          1.000000
     tick         0.980166
     blank        0.932702
     dtype: float64)



### Question 5

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.

Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
def answer_five():
    
    tf = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_tf = tf.transform(X_train)
    X_test_tf = tf.transform(X_test)
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_tf, y_train)
    pred = clf.predict(X_test_tf)
    return roc_auc_score(y_test, pred)
```


```python
answer_five()
```




    0.94162436548223349



### Question 6

What is the average length of documents (number of characters) for not spam and spam documents?

*This function should return a tuple (average length not spam, average length spam).*


```python
def answer_six():
    
    len_spam = [len(x) for x in spam_data.loc[spam_data['target']==1, 'text']]
    len_not_spam = [len(x) for x in spam_data.loc[spam_data['target']==0, 'text']]
    return (np.mean(len_not_spam), np.mean(len_spam))
```


```python
answer_six()
```




    (71.023626943005183, 138.8661311914324)



<br>
<br>
The following function has been provided to help you combine new features into the training data:


```python
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
```

### Question 7

Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.

Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.svm import SVC

def answer_seven():
    
    len_train = [len(x) for x in X_train]
    len_test = [len(x) for x in X_test]
    
    tf = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_tf = tf.transform(X_train)
    X_test_tf = tf.transform(X_test)
    
    X_train_tf = add_feature(X_train_tf, len_train)
    X_test_tf = add_feature(X_test_tf, len_test)
    
    clf = SVC(C=10000)
    clf.fit(X_train_tf, y_train)
    pred = clf.predict(X_test_tf)
    
    return roc_auc_score(y_test, pred)
```


```python
answer_seven()
```




    0.95813668234215565



### Question 8

What is the average number of digits per document for not spam and spam documents?

*This function should return a tuple (average # digits not spam, average # digits spam).*


```python
def answer_eight():
    
    dig_spam = [sum(char.isnumeric() for char in x) for x in spam_data.loc[spam_data['target']==1,'text']]
    dig_not_spam = [sum(char.isnumeric() for char in x) for x in spam_data.loc[spam_data['target']==0,'text']]
    return (np.mean(dig_not_spam), np.mean(dig_spam))
```


```python
answer_eight()
```




    (0.29927461139896372, 15.76037483266399)



### Question 9

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* **number of digits per document**

fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    dig_train = [sum(char.isnumeric() for char in x) for x in X_train]
    dig_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
    tf = TfidfVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
    X_train_tf = tf.transform(X_train)
    X_test_tf = tf.transform(X_test)
    
    X_train_tf = add_feature(X_train_tf, dig_train)
    X_test_tf = add_feature(X_test_tf, dig_test)
    
    clf = LogisticRegression(C=100).fit(X_train_tf, y_train)
    pred = clf.predict(X_test_tf)
    
    return roc_auc_score(y_test, pred)
```


```python
answer_nine()
```




    0.96787090640544626



### Question 10

What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?

*Hint: Use `\w` and `\W` character classes*

*This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*


```python
def answer_ten():
    
    return (np.mean(spam_data.loc[spam_data['target']==0,'text'].str.count('\W')), 
            np.mean(spam_data.loc[spam_data['target']==1,'text'].str.count('\W')))
```


```python
answer_ten()
```




    (17.291813471502589, 29.041499330655956)



### Question 11

Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**

To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* number of digits per document
* **number of non-word characters (anything other than a letter, digit or underscore.)**

fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.

Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.

The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.

The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
['length_of_doc', 'digit_count', 'non_word_char_count']

*This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*


```python
def answer_eleven():
    
    len_train = [len(x) for x in X_train]
    len_test = [len(x) for x in X_test]
    dig_train = [sum(char.isnumeric() for char in x) for x in X_train]
    dig_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
    # Not alpha numeric:
    nan_train = X_train.str.count('\W')
    nan_test = X_test.str.count('\W')
    
    cv = CountVectorizer(min_df = 5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
    X_train_cv = cv.transform(X_train)
    X_test_cv = cv.transform(X_test)
    
    X_train_cv = add_feature(X_train_cv, [len_train, dig_train, nan_train])
    X_test_cv = add_feature(X_test_cv, [len_test, dig_test, nan_test])
    
    clf = LogisticRegression(C=100).fit(X_train_cv, y_train)
    pred = clf.predict(X_test_cv)
    
    score = roc_auc_score(y_test, pred)
    
    feature_names = np.array(cv.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef_index = clf.coef_[0].argsort()
    small_coeffs = list(feature_names[sorted_coef_index[:10]])
    large_coeffs = list(feature_names[sorted_coef_index[:-11:-1]])
    
    return (score, small_coeffs, large_coeffs)
```


```python
answer_eleven()
```




    (0.97885931107074342,
     ['. ', '..', '? ', ' i', ' y', ' go', ':)', ' h', 'go', ' m'],
     ['digit_count', 'ne', 'ia', 'co', 'xt', ' ch', 'mob', ' x', 'ww', 'ar'])


