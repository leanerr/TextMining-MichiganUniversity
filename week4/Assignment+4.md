
---

_You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._

---

# Assignment 4 - Document Similarity & Topic Modelling

## Part 1 - Document Similarity

For the first part of this assignment, you will complete the functions `doc_to_synsets` and `similarity_score` which will be used by `document_path_similarity` to find the path similarity between two documents.

The following functions are provided:
* **`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. You will need to use this function in `doc_to_synsets`.
* **`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.

You will need to finish writing the following functions:
* **`doc_to_synsets:`** returns a list of synsets in document. This function should first tokenize and part of speech tag the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it should find each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset match should be used. If there is no match, that token is skipped.
* **`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together and normalize this value by dividing it by the number of largest similarity values found. Be careful with data types, which should be floats. Missing values should be ignored.

Once `doc_to_synsets` and `similarity_score` have been completed, submit to the autograder which will run `test_document_path_similarity` to test that these functions are running correctly. 

*Do not modify the functions `convert_tag`, `document_path_similarity`, and `test_document_path_similarity`.*


```python
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import pandas as pd

# Need to feed pos tags to this function!
def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v' }
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    
    tokens = nltk.word_tokenize(doc)
   #print(tokens)
    pos_tags = nltk.pos_tag(tokens)
   #print(pos_tags)
    wn_tags = [convert_tag(x[1]) for x in pos_tags]
  # print('wn+tags'+ str(wn_tags))
    # If there is nothing in the synset for the token, it must be skipped! Therefore check that len of the synset is > 0!
    # Will return a list of lists of synsets - one list for each token!
    # Remember to use only the first match for each token! Hence wn.synsets(x,y)[0]!
    synset_list = [wn.synsets(x,y)[0] for x,y in zip(tokens, wn_tags) if len(wn.synsets(x,y))>0]
    return synset_list


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    max_sim = []
    for syn in s1:
        sim = [syn.path_similarity(x) for x in s2 if syn.path_similarity(x) is not None]
        if sim:
            max_sim.append(max(sim))
    return np.mean(max_sim)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    res = similarity_score(synsets1, synsets2)+similarity_score(synsets2, synsets1)/2
    return res
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /home/jovyan/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /home/jovyan/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!



```python
synsets1 = doc_to_synsets('I like cats')
synsets2 = doc_to_synsets('I like dogs')
similarity_score(synsets1, synsets2)
```

    ['I', 'like', 'cats']
    [('I', 'PRP'), ('like', 'VBP'), ('cats', 'NNS')]
    wn+tags[None, 'v', 'n']
    ['I', 'like', 'dogs']
    [('I', 'PRP'), ('like', 'VBP'), ('dogs', 'NNS')]
    wn+tags[None, 'v', 'n']





    0.73333333333333339




```python
synsets1 = doc_to_synsets('I like cats')
synsets2 = doc_to_synsets('I like dogs')
similarity_score(synsets1, synsets2)+similarity_score(synsets2, synsets1)/2
```

    ['I', 'like', 'cats']
    [('I', 'PRP'), ('like', 'VBP'), ('cats', 'NNS')]
    wn+tags[None, 'v', 'n']
    ['I', 'like', 'dogs']
    [('I', 'PRP'), ('like', 'VBP'), ('dogs', 'NNS')]
    wn+tags[None, 'v', 'n']





    1.1000000000000001




```python
doc_to_synsets('Fish are nvqjp friends.')

```

    ['Fish', 'are', 'nvqjp', 'friends', '.']
    [('Fish', 'NN'), ('are', 'VBP'), ('nvqjp', 'JJ'), ('friends', 'NNS'), ('.', '.')]
    wn+tags['n', 'v', 'a', 'n', None]





    [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]




```python
doc_to_synsets('I like cats')
```

    ['I', 'like', 'cats']
    [('I', 'PRP'), ('like', 'VBP'), ('cats', 'NNS')]
    wn+tags[None, 'v', 'n']





    [Synset('one.n.01'), Synset('like.v.02'), Synset('guy.n.01')]




```python
document_path_similarity('I am Ali','She is mikasa')
```

    ['I', 'am', 'Ali']
    [('I', 'PRP'), ('am', 'VBP'), ('Ali', 'RB')]
    wn+tags[None, 'v', 'r']
    ['She', 'is', 'mikasa']
    [('She', 'PRP'), ('is', 'VBZ'), ('mikasa', 'JJ')]
    wn+tags[None, 'v', 'a']





    1.5



### test_document_path_similarity

Use this function to check if doc_to_synsets and similarity_score are correct.

*This function should return the similarity score as a float.*


```python
def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
```


```python
test_document_path_similarity()

```

    ['This', 'is', 'a', 'function', 'to', 'test', 'document_path_similarity', '.']
    [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('function', 'NN'), ('to', 'TO'), ('test', 'VB'), ('document_path_similarity', 'NN'), ('.', '.')]
    wn+tags[None, 'v', None, 'n', None, 'v', 'n', None]
    ['Use', 'this', 'function', 'to', 'see', 'if', 'your', 'code', 'in', 'doc_to_synsets', 'and', 'similarity_score', 'is', 'correct', '!']
    [('Use', 'VB'), ('this', 'DT'), ('function', 'NN'), ('to', 'TO'), ('see', 'VB'), ('if', 'IN'), ('your', 'PRP$'), ('code', 'NN'), ('in', 'IN'), ('doc_to_synsets', 'NNS'), ('and', 'CC'), ('similarity_score', 'NN'), ('is', 'VBZ'), ('correct', 'JJ'), ('!', '.')]
    wn+tags['v', None, 'n', None, 'v', None, None, 'n', None, 'n', None, 'n', 'v', 'a', None]





    0.86051587301587307



<br>
___
`paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.

`Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).


```python
# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()
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
      <th>Quality</th>
      <th>D1</th>
      <th>D2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Ms Stewart, the chief executive, was not expec...</td>
      <td>Ms Stewart, 61, its chief executive officer an...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>After more than two years' detention under the...</td>
      <td>After more than two years in detention by the ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>"It still remains to be seen whether the reven...</td>
      <td>"It remains to be seen whether the revenue rec...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>And it's going to be a wild ride," said Allan ...</td>
      <td>Now the rest is just mechanical," said Allan H...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>The cards are issued by Mexico's consulates to...</td>
      <td>The card is issued by Mexico's consulates to i...</td>
    </tr>
  </tbody>
</table>
</div>



___

### most_similar_docs

Using `document_path_similarity`, find the pair of documents in paraphrases which has the maximum similarity score.

*This function should return a tuple `(D1, D2, similarity_score)`*


```python
def most_similar_docs():
    
    sim_scores = [document_path_similarity(x,y) for x,y in zip(paraphrases['D1'], paraphrases['D2'])]
    return (paraphrases.loc[np.argmax(sim_scores),'D1'], paraphrases.loc[np.argmax(sim_scores),'D2'], max(sim_scores))
```


```python
most_similar_docs()

```




    ('"Indeed, Iran should be put on notice that efforts to try to remake Iraq in their image will be aggressively put down," he said.',
     '"Iran should be on notice that attempts to remake Iraq in Iran\'s image will be aggressively put down," he said.\n',
     1.4506172839506173)



### label_accuracy

Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.

*This function should return a float.*


```python
def label_accuracy():
    from sklearn.metrics import accuracy_score

    paraphrases['sim_scores'] = [document_path_similarity(x,y) for x,y in zip(paraphrases['D1'], paraphrases['D2'])]
    paraphrases['sim_scores'] = np.where(paraphrases['sim_scores']>0.75, 1, 0)
    return accuracy_score(paraphrases['Quality'], paraphrases['sim_scores'])
```


```python
def label_accuracy():
    from sklearn.metrics import accuracy_score

    # Your Code Here
    def func(x):
        try:
            return document_path_similarity(x['D1'], x['D2'])
        except:
            return np.nan

    paraphrases['similarity_score'] = paraphrases.apply(func, axis=1)
    df = paraphrases
    df2 = df.dropna()
    df2['label'] = df2['similarity_score'].apply(lambda x: 1 if x > 0.75 else 0)


    output = accuracy_score(df2['label'], df2['Quality'])

    return output
```


```python
label_accuracy()

```




    0.5



## Part 2 - Topic Modelling

For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`. You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters on the corpus, and save to the variable `ldamodel`. Extract 10 topics using `corpus` and `id_map`, and with `passes=25` and `random_state=34`.


```python
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

```


```python
# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=id_map, passes=25, random_state=34)
```

### lda_topics

Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as a list of 10 tuples where each tuple takes on the form:

`(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')`

for example.

*This function should return a list of tuples.*


```python
def lda_topics():
    
    return list(ldamodel.show_topics(num_topics=10, num_words=10))
```


```python
lda_topics()

```




    [(0,
      '0.056*"edu" + 0.043*"com" + 0.033*"thanks" + 0.022*"mail" + 0.021*"know" + 0.020*"does" + 0.014*"info" + 0.012*"monitor" + 0.010*"looking" + 0.010*"don"'),
     (1,
      '0.024*"ground" + 0.018*"current" + 0.018*"just" + 0.013*"want" + 0.013*"use" + 0.011*"using" + 0.011*"used" + 0.010*"power" + 0.010*"speed" + 0.010*"output"'),
     (2,
      '0.061*"drive" + 0.042*"disk" + 0.033*"scsi" + 0.030*"drives" + 0.028*"hard" + 0.028*"controller" + 0.027*"card" + 0.020*"rom" + 0.018*"floppy" + 0.017*"bus"'),
     (3,
      '0.023*"time" + 0.015*"atheism" + 0.014*"list" + 0.013*"left" + 0.012*"alt" + 0.012*"faq" + 0.012*"probably" + 0.011*"know" + 0.011*"send" + 0.010*"months"'),
     (4,
      '0.025*"car" + 0.016*"just" + 0.014*"don" + 0.014*"bike" + 0.012*"good" + 0.011*"new" + 0.011*"think" + 0.010*"year" + 0.010*"cars" + 0.010*"time"'),
     (5,
      '0.030*"game" + 0.027*"team" + 0.023*"year" + 0.017*"games" + 0.016*"play" + 0.012*"season" + 0.012*"players" + 0.012*"win" + 0.011*"hockey" + 0.011*"good"'),
     (6,
      '0.017*"information" + 0.014*"help" + 0.014*"medical" + 0.012*"new" + 0.012*"use" + 0.012*"000" + 0.012*"research" + 0.011*"university" + 0.010*"number" + 0.010*"program"'),
     (7,
      '0.022*"don" + 0.021*"people" + 0.018*"think" + 0.017*"just" + 0.012*"say" + 0.011*"know" + 0.011*"does" + 0.011*"good" + 0.010*"god" + 0.009*"way"'),
     (8,
      '0.034*"use" + 0.023*"apple" + 0.020*"power" + 0.016*"time" + 0.015*"data" + 0.015*"software" + 0.012*"pin" + 0.012*"memory" + 0.012*"simms" + 0.012*"port"'),
     (9,
      '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')]



### topic_distribution

For the new document `new_doc`, find the topic distribution. Remember to use vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.

*This function should return a list of tuples, where each tuple is `(#topic, probability)`*


```python
new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]
```


```python
def topic_distribution():
    
    sparse_doc = vect.transform(new_doc)
    gen_corpus = gensim.matutils.Sparse2Corpus(sparse_doc, documents_columns=False)
    return list(ldamodel[gen_corpus])[0] # It's a list of lists! You just want the first one.
    #return list(ldamodel.show_topics(num_topics=10, num_words=10)) # For topic_n
```


```python
topic_distribution()

```




    [(0, 0.020001831645100488),
     (1, 0.020002048685728527),
     (2, 0.020000000832306287),
     (3, 0.49628313611665115),
     (4, 0.020002764758471937),
     (5, 0.020002856656041564),
     (6, 0.020001696742412246),
     (7, 0.02000136777340256),
     (8, 0.020001847986127037),
     (9, 0.34370244880375816)]



### topic_names

From the list of the following given topics, assign topic names to the topics you found. If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.

Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, Society & Lifestyle, Religion, Education.

*This function should return a list of 10 strings.*


```python
def topic_names():
    
    output = ['Computers & IT', 'Automobiles', 'Computers & IT', 'Religion', 'Automobiles', 'Sports',
             'Education', 'Religion', 'Computers & IT', 'Science']
    
    return output    

topic_names()
```




    ['Computers & IT',
     'Automobiles',
     'Computers & IT',
     'Religion',
     'Automobiles',
     'Sports',
     'Education',
     'Religion',
     'Computers & IT',
     'Science']


