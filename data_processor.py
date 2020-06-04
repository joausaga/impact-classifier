import nltk

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


nltk.download("stopwords")
nltk.download('wordnet')


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()


def clean_sentence(sentence):
    # Remove HTML tags
    text = BeautifulSoup(sentence, "html.parser").get_text()    
    # Remove non alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Remove leading and trailing spaces
    text = text.strip()
    # Remove extra spaces
    text = ' '.join(text.split()) 
    return text


def sentence_to_words(sentence, context_words=[], lemmatization=False):
    """
    Tokenize, remove stop english words, and steem.
    Words composed of a single character are discarded.
    If required lemmatization is also performed
    
    """
    stop_words = context_words + stopwords.words('english')
    # Split string into words
    words = word_tokenize(sentence)
    # Remove stopwords and put words to lowercase
    words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 1]
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


def count_pos_tag(sentence, tag):
    """
    Count occurrences of tag in sentence
    """
    pos_family = {
        'noun' : ['NN','NNS','NNP','NNPS'],
        'pron' : ['PRP','PRP$','WP','WP$'],
        'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
        'adj' :  ['JJ','JJR','JJS'],
        'adv' : ['RB','RBR','RBS','WRB']
    }
    tag_counter = 0
    sentence_pos_tags = pos_tag(word_tokenize(sentence))    
    for s_pos_tag in sentence_pos_tags:
        s_tag = s_pos_tag[1]
        if s_tag[:2] in pos_family[tag]:
            tag_counter += 1
    return tag_counter


def analyze_sentiment(sentence):
    score = analyzer.polarity_scores(sentence)
    return score['compound']