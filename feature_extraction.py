import warnings
warnings.filterwarnings("ignore")

import os
import json
import glob
import codecs
import time
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
eng_stopwords = set(stopwords.words("english"))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def represent_text(text,n):
    # Extracts all character 'n'-grams from  a 'text'
    if n>0:
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
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

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def meta_features_extractor(corpus):
    sent_len = []
    word_len = []
    word_num = []
    single_num = []
    punct_num = []
    tit_num = []
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
        ##average number of  titles words
        tit_num.append(np.mean(list(map(
            lambda x: len([t for t in str(x) if t.istitle()]), sent_tokenize(paragraph)))))
        ##number of stopwords
        stop_num.append(np.mean(list(map(
            lambda x: len([t for t in str(x) if t in eng_stopwords]), sent_tokenize(paragraph)))))
        ## Number of upper words in the text ##
        upper_num.append(np.mean(list(map(
            lambda x: len([t for t in str(x) if t.isupper()]), sent_tokenize(paragraph)))))

        x = np.array([sent_len, word_len, word_num, single_num, punct_num, tit_num, stop_num, upper_num])

    return x.T

def word_embedding():
    embeddings_index = {}
    f = open('input/glove.840B.300d.txt',encoding="utf8")
    for line in tqdm(f):
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

#I decided to transform whole article to a vector
def sent2vec(corpus):
    words = str(corpus).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in eng_stopwords]#delete high frequency words
    words = [w for w in words if w.isalpha] # only alpha
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def bulid_LR(xtrain, xvalid, ytrain, yvalid, xtest):
    log = LogisticRegression()

    log_parameters = {'C': np.arange(1, 5, 2), 'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
    log_clf = GridSearchCV(log, log_parameters, cv=5, n_jobs=-1)  # 寻找最佳参数，这会很慢，如果不想使用，可以自己修改：把log_clf改为log
    # log.fit(xtrain, ytrain)
    log_clf.fit(xtrain, ytrain)
    print(log_clf.best_params_)
    log_model = log_clf.best_estimator_
    loss = multiclass_logloss(yvalid, log_model.predict_proba(xvalid))  # 模型的loss
    result = log_model.predict(xtest)

    return result, loss


def bulid_SVM(xtrain, xvalid, ytrain, yvalid, xtest):
    svc = SVC()
    svc.probability = True

    svc_parameters = {'kernel': ('linear', 'rbf'), 'C': np.arange(1, 10, 2), 'gamma': np.arange(0.125, 4, 0.5)}
    svc_clf = GridSearchCV(svc, svc_parameters, cv=5, n_jobs=-1)

    svc_clf.fit(xtrain, ytrain)
    print(svc_clf.best_params_)
    svc_model = svc_clf.best_estimator_
    loss = multiclass_logloss(yvalid, svc_model.predict_proba(xvalid))
    result = svc_model.predict(xtest)

    return result, loss


def bulid_RF(xtrain, xvalid, ytrain, yvalid, xtest):
    rf = RandomForestClassifier()

    rf_parameters = {'n_estimators': np.arange(35, 50, 3), 'max_depth': np.arange(4, 9, 2),
                     'min_samples_split': np.arange(30, 50, 5),
                     'min_samples_leaf': np.arange(1, 15, 3), 'max_features': np.arange(0.2, 1, 0.2)}
    rf_clf = GridSearchCV(rf, rf_parameters, cv=5, n_jobs=-1)

    rf_clf.fit(xtrain, ytrain)
    print(rf_clf.best_params_)
    rf_model = rf_clf.best_estimator_
    loss = multiclass_logloss(yvalid, rf_model.predict_proba(xvalid))
    result = rf_model.predict(xtest)

    return result, loss


def bulid_xgb(xtrain, xvalid, ytrain, yvalid, xtest):
    xgb_clf = xgb.XGBClassifier(nthread=10, learning_rate=0.1)

    xgb_parameters = {'max_depth': np.arange(1, 9, 2), 'n_estimators': np.arange(1, 301, 100),
                      'colsample_bytree': np.arange(0.3, 1, 0.3), }
    xgb_Gclf = GridSearchCV(xgb_clf, xgb_parameters, cv=5, n_jobs=-1)

    xgb_Gclf.fit(xtrain, ytrain)
    print(xgb_Gclf.best_params_)
    xgb_model = xgb_Gclf.best_estimator_
    loss = multiclass_logloss(yvalid, xgb_model.predict_proba(xvalid))
    result = xgb_model.predict(xtest)
    return result, loss

def problemset():
    train = all_train_texts[problemsetting]
    test = all_test_texts[problemsetting]
    lab_en = preprocessing.LabelEncoder()
    train_labels = lab_en.fit_transform(all_labels[problemsetting])
    return train,test,train_labels

    # meta features
def meta_features():
    #done
    sent_len = []
    sent_len.append(np.mean(list(map(
                lambda x: len(x.split()), sent_tokenize(train1[0])))))
    meta_train = meta_features_extractor(train1)
    meta_testX = meta_features_extractor(test1)
    meta_trainX, meta_validX, meta_trainY, meta_validY = train_test_split(meta_train, labels1,
                                                      stratify=labels1,
                                                      random_state=42,
                                                      test_size=0.3, shuffle=True)
    return meta_trainX, meta_validX, meta_trainY, meta_validY, meta_testX

def tfidffeature():
    # done
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    all_tfidf = tfv.fit_transform(train1)
    tfv_testX = tfv.transform(test1)
    tfv_trainX, tfv_validX, tfv_trainY, tfv_validY = train_test_split(all_tfidf, labels1,
                                                                      stratify=labels1,
                                                                      random_state=42,
                                                                      test_size=0.3, shuffle=True)
    return tfv_trainX, tfv_validX, tfv_trainY, tfv_validY, tfv_testX

def countfeature():
    #done
    ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), stop_words = 'english')
    all_ctv = ctv.fit_transform(train1)
    ctv_testX = ctv.transform(test1)
    ctv_trainX, ctv_validX, ctv_trainY, ctv_validY = train_test_split(all_ctv, labels1,
                                                                      stratify=labels1,
                                                                      random_state=42,
                                                                      test_size=0.3, shuffle=True)
    return ctv_trainX, ctv_validX, ctv_trainY, ctv_validY, ctv_testX

def wordembeddingfeature():
    embeddings_index = word_embedding() # where to use embeddings_index ?
    words_vector = np.array([sent2vec(x) for x in train1])
    embedding_testX = np.array([sent2vec(x) for x in test1])
    embedding_trainX, embedding_validX, embedding_trainY, embedding_validY = train_test_split(words_vector, labels1,
                                                      stratify=labels1,
                                                      random_state=42,
                                                      test_size=0.3, shuffle=True)
    return embedding_trainX, embedding_validX, embedding_trainY, embedding_validY, embedding_testX

if __name__ == '__main__':
    start_time = time.time()
    path = 'input/pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02/'
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    all_train_texts = []
    all_labels = []
    all_test_texts = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
    for index, problem in enumerate(problems):
        print(problem)
        # Reading information about the problem
        infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training and test set
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files(path + os.sep + problem, candidate))
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        test_docs = read_files(path + os.sep + problem, unk_folder)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', len(test_texts), 'unknown texts')
        all_train_texts.append(
            train_texts)  # all_train_texts[0]-problem0001-len-140, all_train_texts[1]-problem0002-len-35
        all_labels.append(train_labels)  # all_labels[0]-problem0001-len-140, all_labels[1]-problem0002-len-35
        all_test_texts.append(test_texts)  # all_test_texts[0]-problem0001-len-105, all_test_texts[1]-problem0002-len-21

        problemsetting = int(problem[-1])-1  # o or 1, changing this for changing the problem sets
        train1,test1,labels1 = problemset()

        # text based features
        xtrain, xvalid, ytrain, yvalid = train_test_split(train1, labels1,
                                                          stratify=labels1,
                                                          random_state=42,
                                                          test_size=0.3, shuffle=True)

        # the names of test set: meta_testX, tfv_testX, ctv_testX, embedding_testX
        # different features:
        meta_trainX, meta_validX, meta_trainY, meta_validY, meta_testX = meta_features()
        # tfv_trainX, tfv_validX, tfv_trainY, tfv_validY, tfv_testX = tfidffeature()
        # ctv_trainX, ctv_validX, ctv_trainY, ctv_validY, ctv_testX = countfeature()
        # embedding_trainX, embedding_validX, embedding_trainY, embedding_validY, embedding_testX = wordembeddingfeature()


        # now feed them into different models
        #LR_meta_testY, LR_meta_loss = bulid_LR(meta_trainX, meta_validX, meta_trainY, meta_validY, meta_testX)
        SVM_meta_testY, SVM_meta_loss = bulid_SVM(meta_trainX, meta_validX, meta_trainY, meta_validY, meta_testX)
        #RF_meta_testY, RF_meta_loss = bulid_RF(meta_trainX, meta_validX, meta_trainY, meta_validY, meta_testX)
        #xgb_meta_testY, xgb_meta_loss = bulid_xgb(meta_trainX, meta_validX, meta_trainY, meta_validY, meta_testX)

        predictions = ['candidate000{0:0=2d}'.format(a) for a in SVM_meta_testY] # try one of the predictions to evaluate

        # Writing output file
        outpath = 'output/answers'
        out_data = []
        unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        # with using different features and ml methods, 'embedding-svm-' should be replaced by different feature and ml method combinations, inorder to store the file
        with open(outpath + os.sep + 'answers-' + 'embedding-svm-' + problem + '.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        # with using different features and ml methods, 'embedding-svm-' should be replaced by different feature and ml method combinations
        print('\t', 'answers saved to file', 'answers-' + 'embedding-svm-' + problem + '.json')
    print('elapsed time:', time.time() - start_time)




