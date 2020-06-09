from text_processor import sentence_to_words, count_pos_tag, analyze_sentiment
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import get_config

import numpy as np
import io
import joblib
import os
import pandas as pd
import pdftotext
import streamlit as st


def process_sentence(sentence, vectorizer, vectorizer_type):
    dict_process = {
        'char_count': len(sentence),
        'word_count': len(sentence.split()),
        'upper_case_word_count': len([wrd for wrd in sentence.split() if wrd.isupper()]),
        'title_word_count': len([wrd for wrd in sentence.split() if wrd.istitle()]),
        'noun_count': count_pos_tag(sentence, 'noun'),
        'verb_count': count_pos_tag(sentence, 'verb'),
        'adj_count': count_pos_tag(sentence, 'adj'),
        'adv_count': count_pos_tag(sentence, 'adv'),
        'pron_count': count_pos_tag(sentence, 'pron'),
        'sentiment_score': analyze_sentiment(sentence)
    }
    dict_process['word_density'] = dict_process['char_count']/(dict_process['word_count']+1)
    text_features = []
    for _, v in dict_process.items():
        text_features.append(v)
    text_features_array = np.array([text_features])
    sentence_words = sentence_to_words(sentence)
    if vectorizer_type == 'tc':
        bag_of_words = vectorizer.transform([sentence_words]).toarray()
    elif vectorizer_type == 'tfidf':
        bag_of_words = vectorizer.fit_transform([sentence_words]).toarray()
    else:
        raise Exception(f'Unknown vectorizer {vectorizer_type}')
    features = np.concatenate((text_features_array, bag_of_words), axis=1)
    return features


def process_pdf(pdf_file, classifier, vectorizer, vectorizer_type):
    pdf = pdftotext.PDF(pdf_file)
    num_pages = len(pdf)    
    for page_num in range(0, num_pages):
        st.write(f'Page: {page_num+1}')
        raw_sentences = sent_tokenize(pdf[page_num])
        for raw_sentence in raw_sentences:
            processed_sentence = process_sentence(raw_sentence, vectorizer, 
                                                  vectorizer_type)            
            prediction = classifier.predict(processed_sentence)
            if prediction == 1:
                st.write(raw_sentences[0])


def init(model_dir, model_name, data_dir):
    # Load model
    model_dict = joblib.load(os.path.join(model_dir, model_name))
    classifier = model_dict['model']
    # Instantiate vectorizer
    with open(os.path.join(data_dir, 'train', model_dict['data_fn']), 'rb') as f:
        data = joblib.load(f)
    if data['transformation'] == 'tc':
        vectorizer = CountVectorizer(max_features=int(data['max_features']),
                                     ngram_range=data['ngram_range'],
                                     vocabulary=data['vocabulary'],
                                     preprocessor=lambda x: x, tokenizer=lambda x: x,
                                     lowercase=False, analyzer='word',
                                     token_pattern=r'\w{1,}')
    elif data['transformation'] == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=int(data['max_features']), 
                                     ngram_range=data['ngram_range'],
                                     vocabulary=data['vocabulary'],
                                     preprocessor=lambda x: x, 
                                     tokenizer=lambda x: x,
                                     lowercase=False, analyzer='word', 
                                     token_pattern=r'\w{1,}')
    else:
        msg = 'Unknown transformation: {}'.format(data['transformation'])
        raise Exception(msg)

    return classifier, vectorizer, data['transformation']


def main():
    app_config = get_config('config.json')
    model_dir = app_config['model_dir']
    model_name = app_config['model_name']
    data_dir = app_config['data_dir']

    classifier, vectorizer, vectorizer_type = init(model_dir, model_name, 
                                                   data_dir)
    st.title('Impact Classifier')
    st.write("""
    Impact Classifier is a machine-learning-based document classifier that
    automatically identifies evidence of social impact in research documents.
    """)
    uploaded_file = st.file_uploader('Select a file to analyze', type='pdf')
    if uploaded_file is not None:
        process_pdf(uploaded_file, classifier, vectorizer, vectorizer_type)


if __name__ == "__main__":
    main()