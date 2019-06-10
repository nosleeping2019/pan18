# -*- coding: utf-8 -*-

"""
 A baseline authorship attribution method
 based on a character n-gram representation
 and a linear SVM classifier
 for Python 2.7
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-18 cross-domain authorship attribution task
 See details here: http://pan.webis.de/clef18/pan18-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
 - scikit-learn

 Usage from command line: 
    > python pan18-cdaa-baseline.py -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY [-n N-GRAM-ORDER] [-ft FREQUENCY-THRESHOLD] [-c CLASSIFIER]
 EVALUATION-DIRECTORY (str) is the main folder of a PAN-18 collection of attribution problems
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-18 format
 Optional parameters of the model:
   N-GRAM-ORDER (int) is the length of character n-grams (default=3)
   FREQUENCY-THRESHOLD (int) is the curoff threshold used to filter out rare n-grams (default=5)
   CLASSIFIER (str) is either 'OneVsOne' or 'OneVsRest' version of SVM (default=OneVsRest)
   
 Example:
     > python pan18-cdaa-baseline.py -i "mydata/pan18-cdaa-development-corpus" -o "mydata/pan18-answers"
"""

from __future__ import print_function
import os
import glob
import json
import argparse
import time
import codecs
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from string import punctuation
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
from nltk.tokenize import sent_tokenize, word_tokenize


def represent_text(text,n):
    # Extracts all character 'n'-grams from  a 'text'
    tokens = []
    if n>0: #extract n-gram
        tokens_n = [text[i:i+n] for i in range(len(text)-n+1)]
        tokens.extend(tokens_n)
    if n-1 > 0: # extract n-1-gram
        tokens_n_1 = [text[i:i+n-1] for i in range(len(text)-n+1)]
        tokens.extend(tokens_n_1)
    if n-2 > 0: # extract n-2-gram
        tokens_n_2 = [text[i:i + n - 2] for i in range(len(text) - n + 1)]
        tokens.extend(tokens_n_2)
    if n-3 > 0: # extract n-3-gram
        tokens_n_3 = [text[i:i + n - 3] for i in range(len(text) - n + 1)]
        tokens.extend(tokens_n_3)
    if n-4 > 0: # extract n-4-gram
        tokens_n_4 = [text[i:i + n - 4] for i in range(len(text) - n + 1)]
        tokens.extend(tokens_n_4)
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path+os.sep+label+os.sep+'*.txt')
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label))
        f.close()
    return texts

def extract_vocabulary(texts,n,ft):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences=defaultdict(int)
    for (text,label) in texts:
        text_occurrences=represent_text(text,n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def meta_features_extractor(corpus):
    sent_len = []
    word_len = []
    word_num = []
    single_num = []
    punct_num = []
    stop_num = []
    upper_num = []
    for paragraph in corpus:
        ## average lenth of sentences
        sent_len.append(np.mean(list(map(
            lambda x: len(x.split()), sent_tokenize(paragraph)))))
        ## average lenth of words
        word_len.append(np.mean(list(map(
            lambda x: len(str(x)), word_tokenize(paragraph)))))
        ##number of words
        word_num.append(len(word_tokenize(paragraph)))
        ##number of single words
        single_num.append(len([w for w in set(word_tokenize(paragraph)) if w not in punctuation]))
        ## average number of punctuation in a sentence
        punct_num.append(np.mean(list(map(
            lambda x: len([p for p in str(x) if p in punctuation]), sent_tokenize(paragraph)))))
        ##number of stopwords
        stop_num.append(np.mean(list(map(
            lambda x: len([t for t in str(x) if t in eng_stopwords]), sent_tokenize(paragraph)))))
        ## Number of upper words in the text ##
        upper_num.append(np.mean(list(map(
            lambda x: len([t for t in str(x) if t.isupper()]), sent_tokenize(paragraph)))))

        x = np.array([sent_len, word_len, word_num, single_num, punct_num, stop_num, upper_num])

    return x.T

def meta_data(train_texts,test_texts):
    train_data = meta_features_extractor(train_texts)
    train_data = train_data.astype(float)
    for i, v in enumerate(train_texts):
        train_data[i] = train_data[i] / len(train_texts[i])  # normalize
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    test_data = meta_features_extractor(test_texts)
    test_data = test_data.astype(float)
    for i, v in enumerate(test_texts):
        test_data[i] = test_data[i] / len(test_texts[i])  # normalize
    scaler.transform(test_data)
    return scaler, train_data, test_data

def count_data(train_docs, n, ft, train_texts,test_texts):
    vocabulary = extract_vocabulary(train_docs, n, ft)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, lowercase=False, vocabulary=vocabulary)
    train_data = vectorizer.fit_transform(train_texts)
    train_data = train_data.astype(float)
    for i, v in enumerate(train_texts):
        train_data[i] = train_data[i] / len(train_texts[i])  # normalize

    test_data = vectorizer.transform(test_texts)
    test_data = test_data.astype(float)
    for i, v in enumerate(test_texts):
        test_data[i] = test_data[i] / len(test_texts[i])  # normalize
    return vectorizer, vocabulary,train_data, test_data

def tfidf_data(train_docs, n, ft, train_texts,test_texts):
    vocabulary = extract_vocabulary(train_docs, n, ft)
    vectorizer = TfidfVectorizer(min_df=3, max_features=None,
                                 strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                                 ngram_range=ngram_range, use_idf=True, smooth_idf=True, sublinear_tf=True,
                                 stop_words='english', vocabulary = vocabulary)
    train_data = vectorizer.fit_transform(train_texts)
    train_data = train_data.astype(float)
    for i, v in enumerate(train_texts):
        train_data[i] = train_data[i] / len(train_texts[i])  # normalize
    test_data = vectorizer.transform(test_texts)
    test_data = test_data.astype(float)
    for i, v in enumerate(test_texts):
        test_data[i] = test_data[i] / len(test_texts[i])  # normalize
    return vectorizer, vocabulary, train_data, test_data


def baseline(path,outpath,n=3,ft=5,classifier='OneVsRest'):
    start_time = time.time()
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    
    for index,problem in enumerate(problems):
        print(problem)
        # Reading information about the problem
        infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training set
        train_docs=[]
        for candidate in candidates:
            train_docs.extend(read_files(path+os.sep+problem,candidate))
        train_texts = [text for i,(text,label) in enumerate(train_docs)]
        train_labels = [label for i,(text,label) in enumerate(train_docs)]

        # Building test set
        test_docs = read_files(path + os.sep + problem, unk_folder)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]

        if attribute == 'baseline-' : # ngram_range(4,4) (0.571) higher than ngram_range(1,3) (0.568) higher than (3,3) (0.507)
            vectorizer, vocabulary, train_data, test_data = count_data(train_docs,n,ft,train_texts,test_texts)
            print('\t', 'vocabulary size:', len(vocabulary))
        if attribute == 'tfidf-char-': # ngram_range(1,3)  (0.51) lower than (3,3) (0.577)
            vectorizer, vocabulary, train_data, test_data = tfidf_data(train_docs,n,ft,train_texts,test_texts)
            print('\t', 'vocabulary size:', len(vocabulary))
        if attribute == 'meta-':
            scaler, train_data, test_data = meta_data(train_texts,test_texts)
            print('\t', 'meta feature dimension:', len(train_data))

        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', len(test_texts), 'unknown texts')

        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        if classifier=='OneVsOne':
            clf=OneVsOneClassifier(LinearSVC(C=1)).fit(scaled_train_data, train_labels)
        else:
            clf=OneVsRestClassifier(LinearSVC(C=1)).fit(scaled_train_data, train_labels)
        predictions=clf.predict(scaled_test_data)
        print('predictions',predictions)
        # Writing output file
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath+os.sep+'answers-'+ attribute + ml + ngram_range_name + problem +'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        print('\t', 'answers saved to file','answers-' + attribute + ml + ngram_range_name + problem +'.json')
    print('elapsed time:', time.time() - start_time)

def main():
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='PAN-18 Baseline Authorship Attribution Method')
    # parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    # parser.add_argument('-o', type=str, help='Path to an output folder')
    # parser.add_argument('-n', type=int, default=3, help='n-gram order (default=3)')
    # parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    # parser.add_argument('-c', type=str, default='OneVsRest', help='OneVsRest or OneVsOne (default=OneVsRest)')
    # args = parser.parse_args()
    # if not args.i:
    #     print('ERROR: The input folder is required')
    #     parser.exit(1)
    # if not args.o:
    #     print('ERROR: The output folder is required')
    #     parser.exit(1)
    # baseline(args.i, args.o, args.n, args.ft, args.c)

    path = 'input/pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/'
    outpath = 'output/answers'
    baseline(path,outpath,4,5)

attributes = ['baseline-', 'tfidf-char-', 'meta-']
ngram_ranges = [None,(1,3),(3,3),(1,4),(4,4)]
ngram_range_names = ['No-gram-','1-3-','3-3-','1-4-','4-4-']
mls = ['baseline-']

attribute = attributes[2]
ngram_range = ngram_ranges[0]
ngram_range_name = ngram_range_names[0]
ml = mls[0]

if __name__ == '__main__':
    main()