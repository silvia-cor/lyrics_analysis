import nltk
import typing
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


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


def _Pos_ngrams(docs_tr, docs_te):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 5))
    X_tr = vectorizer.fit_transform(_POS_tags(docs_tr))
    X_te = vectorizer.transform(_POS_tags(docs_te))
    return X_tr, X_te


def _feature_selection(docs_tr, docs_te, y_tr):
    selector = SelectKBest(chi2, k=100)
    X_tr = selector.fit_transform(docs_tr, y_tr)
    X_te = selector.transform(docs_te)
    return X_tr, X_te


def extract_features_authorship(docs_tr, docs_te, y_tr):
    print(f'----- AUTHORSHIP FEATURE EXTRACTION -----')
    # final matrixes of features
    # initialize the right number of rows, or hstack won't work
    X_tr = csr_matrix((len(docs_tr), 0))
    X_te = csr_matrix((len(docs_te), 0))

    fw = get_function_words('english')

    f = normalize(_function_words_freq(docs_tr, fw))
    X_tr = hstack((X_tr, f))
    f = normalize(_function_words_freq(docs_te, fw))
    X_te = hstack((X_te, f))
    print(f'task function words (#features={f.shape[1]}) [Done]')

    f = normalize(_word_lengths_freq(docs_tr))
    X_tr = hstack((X_tr, f))
    f = normalize(_word_lengths_freq(docs_te))
    X_te = hstack((X_te, f))
    print(f'task word lengths (#features={f.shape[1]}) [Done]')

    f_tr, f_te = _Pos_ngrams(docs_tr, docs_te)
    X_tr = hstack((X_tr, f_tr))
    X_te = hstack((X_te, f_te))
    print(f'task POS-ngrams (#features={f_tr.shape[1]}) [Done]')

    X_tr, X_te = _feature_selection(X_tr, X_te, y_tr)

    return X_tr.toarray(), X_te.toarray()
