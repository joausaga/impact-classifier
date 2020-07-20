from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from text_processor import sentence_to_words, count_pos_tag, analyze_sentiment, \
                           clean_sentence, fix_latin_abbreviations, is_valid_sentence
from utils import get_config

import fitz
import io
import joblib
import numpy as np
import re
import os
import pandas as pd
import pdftotext
import streamlit as st


def process_sentence(sentence, vectorizer, vectorizer_type, lemmatization):
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
    sentence_words = sentence_to_words(sentence, context_words=['reseach', 'uk'], 
                                       lemmatization=lemmatization)
    if vectorizer_type == 'tc':
        vec_features = vectorizer.transform([sentence_words]).toarray()
    elif vectorizer_type == 'tfidf':
        vec_features = vectorizer.fit_transform([sentence_words]).toarray()
    else:
        raise Exception(f'Unknown vectorizer {vectorizer_type}')
    features = np.concatenate((vec_features, text_features_array), axis=1)
    return features


def process_pdf(pdf_file, file_name, classifier, vectorizer, vectorizer_type, 
                lemmatization):    
    pdf = fitz.open(stream=pdf_file, filetype='pdf')
    num_pages = pdf.pageCount
    impact_sentences = []
    p_bar = st.progress(0)
    for page_num in range(0, num_pages):
        page = pdf.loadPage(page_num)
        page_text = page.getText("text")
        page_text = fix_latin_abbreviations(page_text)
        raw_sentences = sent_tokenize(page_text)
        for raw_sentence in raw_sentences:
            clean_text = clean_sentence(raw_sentence)
            if is_valid_sentence(clean_text):
                processed_sentence = process_sentence(clean_text, vectorizer, 
                                                      vectorizer_type, 
                                                      lemmatization)            
                prediction = classifier.predict(processed_sentence)
                if prediction[0] == 1:                    
                    if len(impact_sentences) > 0:
                        found_same_text = False
                        for impact_sentence in impact_sentences:
                            if clean_text == impact_sentence['Sentence']:
                                found_same_text = True
                                break
                    if len(impact_sentences) == 0 or not found_same_text:
                        impact_sentences.append(
                            {
                                'File': file_name if file_name else 'file_name', 
                                'Page': page_num+1, 
                                'Sentence': clean_text
                            }
                        )                    
            p_bar.progress(round((page_num)/num_pages,0))
    num_impact_sentences = len(impact_sentences)
    display_msg = f'Found **{num_impact_sentences}** evidence of social impact in the document of {num_pages} pages' 
    if num_impact_sentences == 0:
        st.info(display_msg)
    else:
        st.success(display_msg + '. See table below')
        st.table(impact_sentences)
    

def init(model_dir, model_name, data_dir):
    # Load model
    model_dict = joblib.load(os.path.join(model_dir, model_name))
    classifier = model_dict['model']
    # Instantiate vectorizer
    transformation = model_dict['transformation']
    if transformation['type'] == 'tc':
        vectorizer = CountVectorizer(max_features=int(transformation['max_features']),
                                     ngram_range=transformation['ngram_range'],
                                     vocabulary=transformation['vocabulary'],
                                     preprocessor=lambda x: x, tokenizer=lambda x: x,
                                     lowercase=False, analyzer='word',
                                     token_pattern=r'\w{1,}')
    elif transformation['type'] == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=int(transformation['max_features']), 
                                     ngram_range=transformation['ngram_range'],
                                     vocabulary=transformation['vocabulary'],
                                     preprocessor=lambda x: x, 
                                     tokenizer=lambda x: x,
                                     lowercase=False, analyzer='word', 
                                     token_pattern=r'\w{1,}')
    else:
        msg = 'Unknown transformation: {}'.format(transformation['type'])
        raise Exception(msg)
    if 'lemmatization' in transformation['file_name']:
        lemmatization = True
    else:
        lemmatization = False

    return classifier, vectorizer, transformation['type'], lemmatization


def main():
    app_config = get_config('config.json')
    model_dir = app_config['model_dir']
    model_name = app_config['model_name']
    data_dir = app_config['data_dir']

    classifier, vectorizer, vectorizer_type, lemmatization = \
        init(model_dir, model_name, data_dir)
    st.title('Impact Classifier')
    st.write("""
    Impact Classifier is a machine-learning-based document classifier that
    automatically identifies evidence of social impact in research documents.
    """)
    uploaded_file = st.file_uploader('Select PDF documents to analyze', type='pdf')
    if uploaded_file is not None:
        process_pdf(uploaded_file, '', classifier, vectorizer, vectorizer_type, 
                    lemmatization)


if __name__ == "__main__":
    main()