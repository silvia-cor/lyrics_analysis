import nltk
import numpy as np
import typing
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import sys
import os
from joblib import Parallel, delayed
from tqdm import tqdm


# TODO: feature selection for everything

# tokenize text without punctuation
def tokenize_nopunct(text: str) -> typing.List[str]:
    unmod_tokens = nltk.word_tokenize(text)
    return [token.lower() for token in unmod_tokens if any(char.isalpha() for char in token)]  # checks whether all the chars are alphabetic


# return list of function words
def get_function_words(lang):
    if lang in ['english', 'spanish', 'italian']:
        return stopwords.words(lang)
    else:
        raise ValueError('{} not in scope!'.format(lang))


# extract the frequency (L1x1000) of each function word used in the documents
def _function_words_freq(documents, function_words):
    features = []
    for text in documents:
        mod_tokens = tokenize_nopunct(text)
        freqs = nltk.FreqDist(mod_tokens)
        nwords = len(mod_tokens)
        funct_words_freq = [freqs[function_word] / nwords for function_word in function_words]
        features.append(funct_words_freq)
    f = csr_matrix(features)
    return f


# extract the frequencies (L1x1000) of the words' lengths used in the documents,
# following the idea behind Mendenhall's Characteristic Curve of Composition
def _word_lengths_freq(documents, upto=26):
    features = []
    for text in documents:
        mod_tokens = tokenize_nopunct(text)
        nwords = len(mod_tokens)
        tokens_len = [len(token) for token in mod_tokens]
        tokens_count = []
        for i in range(1, upto):
            tokens_count.append((sum(j >= i for j in tokens_len)) / nwords)
        features.append(tokens_count)
    f = csr_matrix(features)
    return f


def _POS_tags(docs):
    pos_tags = []
    for doc in docs:
        tokens = nltk.word_tokenize(doc)
        tags = nltk.pos_tag(tokens)
        pos_tags.append(' '.join(tag[1] for tag in tags))
    return pos_tags


# Todo: parallelize if you can
def _prosody(docs):
    prosodies = []
    for doc in tqdm(docs):
        for line in doc.splitlines():
            print(line)
            sys.__stdout__ = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            prosody = prosodic.Text(line)
            prosody.parse()
            sub_lines = []
            sys.stdout = sys.__stdout__
            for parse in prosody.bestParses():
                if parse is not None:
                    print(str(parse))
                    sub_lines.append(str(parse))
            prosodies.append(''.join(sub_line for sub_line in sub_lines))
            #lines = [str(parse) for parse in prosody.bestParses()]
    return prosodies


def _ngrams(docs_tr, docs_te, y_tr, analyzer, ngram_range, lowercase=True):
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase)
    X_tr = vectorizer.fit_transform(docs_tr)
    X_te = vectorizer.transform(docs_te)
    X_tr, X_te, selected = _feature_selection(X_tr, X_te, y_tr)
    vocabs = list(vectorizer.vocabulary_.keys())
    feat_names = [vocabs[i] for i in range(len(vectorizer.vocabulary_)) if selected[i]]
    return X_tr, X_te, feat_names


def _feature_selection(X_tr, X_te, y_tr, feature_selection_ratio=0.05):
    num_feats = int(X_tr.shape[1] * feature_selection_ratio)  # number of selected features (must be int)
    selector = SelectKBest(chi2, k=num_feats)
    X_tr = selector.fit_transform(X_tr, y_tr)
    X_te = selector.transform(X_te)
    return X_tr, X_te, selector.get_support()


def extract_features_base(docs_tr, docs_te, y_tr):
    print(f'----- BASE FEATURE EXTRACTION -----')
    feat_names = []

    # final matrixes of features
    # initialize the right number of rows, or hstack won't work
    X_tr = csr_matrix((len(docs_tr), 0))
    X_te = csr_matrix((len(docs_te), 0))

    fw = get_function_words('english')

    f = normalize(_function_words_freq(docs_tr, fw))
    X_tr = hstack((X_tr, f))
    f = normalize(_function_words_freq(docs_te, fw))
    X_te = hstack((X_te, f))
    feat_names = feat_names + fw
    print(f'task function words (#features={f.shape[1]}) [Done]')

    f = normalize(_word_lengths_freq(docs_tr))
    X_tr = hstack((X_tr, f))
    f = normalize(_word_lengths_freq(docs_te))
    X_te = hstack((X_te, f))
    feat_names = feat_names + [i for i in range(1, 26)]
    print(f'task word lengths (#features={f.shape[1]}) [Done]')

    f_tr, f_te, f_names = _ngrams(_POS_tags(docs_tr), _POS_tags(docs_te), y_tr, 'word', (1, 3))
    X_tr = hstack((X_tr, f_tr))
    X_te = hstack((X_te, f_te))
    feat_names = feat_names + f_names
    print(f'task POS-ngrams (#features={f_tr.shape[1]}) [Done]')

    return X_tr.toarray(), X_te.toarray(), feat_names


def extract_features_charngrams(docs_tr, docs_te, y_tr):
    print(f'----- CHAR-NGRAMS FEATURE EXTRACTION -----')
    X_tr, X_te, feat_names = _ngrams(docs_tr, docs_te, y_tr, 'char', (2, 5))
    print(f'task char-ngrams (#features={X_tr.shape[1]}) [Done]')
    return X_tr.toarray(), X_te.toarray(), feat_names


def extract_features_wordngrams(docs_tr, docs_te, y_tr):
    print(f'----- WORD-NGRAMS FEATURE EXTRACTION -----')
    X_tr, X_te, feat_names = _ngrams(docs_tr, docs_te, y_tr, 'word', (1, 3))
    print(f'task word-ngrams (#features={X_tr.shape[1]}) [Done]')
    return X_tr.toarray(), X_te.toarray(), feat_names


def extract_features_prosody(docs_tr, docs_te, y_tr):
    print(f'----- PROSODY-NGRAMS FEATURE EXTRACTION -----')
    X_tr, X_te, feat_names = _ngrams(_prosody(docs_tr), _prosody(docs_te), y_tr, 'char', (2, 5), lowercase=False)
    print(f'task prosody-ngrams (#features={X_tr.shape[1]}) [Done]')
    return X_tr.toarray(), X_te.toarray(), feat_names


# not counting prosody
def extract_features_all(docs_tr, docs_te, y_tr):
    X_tr_base, X_te_base, base_names = extract_features_base(docs_tr, docs_te, y_tr)
    X_tr_charngrams, X_te_charngrams, char_names = extract_features_charngrams(docs_tr, docs_te, y_tr)
    X_tr_wordngrams, X_te_wordngrams, word_names = extract_features_wordngrams(docs_tr, docs_te, y_tr)
    return np.hstack((X_tr_base, X_tr_charngrams, X_tr_wordngrams)), \
           np.hstack((X_te_base, X_te_charngrams, X_te_wordngrams)), \
           base_names + char_names + word_names
