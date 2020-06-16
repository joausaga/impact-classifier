from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import learning_curve

import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import unicodedata


def remove_non_ascii(words):
    """Remove non-ASCII characters from words"""
    words = get_list(words)
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters of words to lowercase"""
    words = get_list(words)
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from words"""
    words = get_list(words)
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_extra_spaces(words):
    words = get_list(words)
    new_words = []
    for word in words:
        word_clean = ' '.join(word.split())
        new_words.append(word_clean)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    words = get_list(words)
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def get_list(text):
    if not isinstance(text, list):
        return word_tokenize(text)
    else:
        return text


def normalize(words):
    words = get_list(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_extra_spaces(words)
    words = remove_stopwords(words)
    return words


def lemmatize_words(words, pos='v'):
    """Lemmatize verbs in list of tokenized words"""
    words = get_list(words)
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        lemmas.append(lemma)
    return lemmas


def save_model(model, name, data_dir, data_filename):
    models_dir = 'models'  # directory to store models
    os.makedirs(models_dir, exist_ok=True)  # ensure the directory exists
    data_fn = data_filename.split('/')[-1]
    with open(os.path.join(data_dir, 'train', data_fn), 'rb') as f:
        data = joblib.load(f)
    transformation = {
        'type': data['transformation'],
        'max_features': data['max_features'],
        'ngram_range': data['ngram_range'],
        'vocabulary': data['vocabulary'],
        'file_name': data_fn
    }
    model_name = f'{name}.joblib'
    model_file_name = os.path.join(models_dir, model_name)
    model_dict = dict(model=model, transformation=transformation)
    joblib.dump(model_dict, model_file_name)


# Get configuration from file
def get_config(config_file):
    with open(str(config_file), 'r') as f:
        config = json.loads(f.read())
    return config


def plot_learning_curve(estimator, title, X, y, metric, ylim=None, cv=None, 
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), 
                        shuffle=False, save_fig=False):
    """
    Generate a simple plot of the test and training learning curve
    Taken from 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    
    plt.figure(figsize=(12,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, 
        n_jobs=n_jobs, cv=cv, train_sizes=train_sizes, shuffle=shuffle, 
        scoring=metric)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")

    if save_fig:
        name_fig_file = f'{title} - {metric}.png'
        if not os.path.exists('figures'):
            os.mkdir('figures')
        fig_path = os.path.join('figures', name_fig_file)
        plt.savefig(fig_path, dpi=200)
    
    return plt


def get_features_and_labels(file_name):
    with open(file_name, 'rb') as f:
        data_dict = joblib.load(f)
    data = data_dict['data']
    labels = data['label'].values
    text_features = list(data.iloc[:,0].values)        
    extra_features = np.array(data.iloc[:,2:].values)
    features = np.concatenate((text_features, extra_features), axis=1)
    return features, labels