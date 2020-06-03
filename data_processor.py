import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup


nltk.download("stopwords")
nltk.download('wordnet')


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def sentence_to_words(sentence, lemmatization=False):
    """
    Remove any html formatting that may appear in sentence as well as 
    tokenize and remove stop english words.
    
    If required steeming and lemmatization is also perform
    
    """
    # Remove HTML tags
    text = BeautifulSoup(sentence, "html.parser").get_text()    
    # Remove non alphabetic characters and put text to lower case
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())    
    # Split string into words
    words = word_tokenize(text)
     # Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]
    # Apply steeming
    words = [stemmer.stem(word) for word in words]
    
    if lemmatization:
        words = [lemmatizer.lemmatize(word) for word in words]
    
    return words


def extract_BoW_features(sentences, max_features=None, transformation='tc', ngram_range=(1,1)):
    """
    Apply transformations to sentences. The transformation
    to by applied are passes as the transformation parameter.
    
    Supported transformations are:
    - tc: term count
    - tfidf: term frequency inverse document frequency
    
    """
    if transformation == 'tc':
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range, 
                                     preprocessor=lambda x: x, tokenizer=lambda x: x,
                                     lowercase=False, analyzer='word', token_pattern=r'\w{1,}')    
    elif transformation == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, 
                                     preprocessor=lambda x: x, tokenizer=lambda x: x,
                                     lowercase=False, analyzer='word', token_pattern=r'\w{1,}')
    bow_features = vectorizer.fit_transform(sentences).toarray()
    
    return bow_features, vectorizer.vocabulary_