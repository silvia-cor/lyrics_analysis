import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import preprocessing.feature_extraction
from preprocessing.preprocessing_classification import genres_acc
import os
import csv
from preprocessing import fetch_dataset
import statistics as stat
from operator import itemgetter


def _balance_for_bin(X, y, unique, seed):
    y = (y == unique).astype(int)  # positive_class = 1; negative_class = 0
    n_pos = np.count_nonzero(y == 1)
    if (n_pos * 2) >= (len(y) - n_pos):
        n_neg = len(y) - n_pos
        print(f'#pos:{n_pos} #neg:{n_neg}')
        return X, y
    else:
        n_neg = n_pos
        print(f'#pos:{n_pos} #neg:{n_neg}')
        np.random.seed(seed)
        to_delete = np.random.choice(np.where(y == 0)[0], len(y)-(n_pos + n_neg), replace=False)
        new_X = np.delete(X, to_delete)
        new_y = np.delete(y, to_delete)
        return new_X, new_y


def kcv_classification(dataset, domain='artists', feat='base'):
    domains = ['artists', 'genres']
    feats = ['author', 'charngrams', 'wordngrams', 'phonetics', 'all']
    assert domain in domains, f'Only possible domain are {domains}'
    assert feat in feats, f'Only possible features are {feats}'
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_all_pred = []
    y_all_te = []
    X = dataset['lyrics']
    y = dataset[domain]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
        X_tr = X[train_index]
        X_te = X[test_index]
        y_tr = y[train_index]
        y_te = y[test_index]
        X_tr, X_te, feat_names = getattr(preprocessing.feature_extraction, f'extract_features_{feat}')(X_tr, X_te, y_tr)
        print("Training shape: ", X_tr.shape)
        print("Test shape: ", X_te.shape)
        print('CLASSIFICATION')
        cls = LinearSVC(class_weight='balanced', random_state=42).fit(X_tr, y_tr)
        y_pred = cls.predict(X_te)
        f1 = f1_score(y_te, y_pred, average='macro')
        print(f'F1: {f1:.3f}')
        y_all_pred.extend(y_pred)
        y_all_te.extend(y_te)

    print(f'----- FINAL RESULTS -----')
    macro_f1 = f1_score(y_all_te, y_all_pred, average='macro')
    micro_f1 = f1_score(y_all_te, y_all_pred, average='micro')
    print(f'Macro-F1: {macro_f1:.3f}')
    print(f'Micro-F1: {micro_f1:.3f}')
    return macro_f1, micro_f1


def binary_classification(domain='artists', feat='base', save=False):
    domains = ['artists', 'genres']
    feats = ['author', 'charngrams', 'wordngrams', 'phonetics', 'all']
    assert domain in domains, f'Only possible domain are {domains}'
    assert feat in feats, f'Only possible features are {feats}'
    dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=False,
                            as_dict=True, random_authors=0)
    X = dataset['lyrics']
    y = dataset[domain]
    uniques = np.unique(y)
    rows = []
    for unique in uniques:
        if domain == 'genres':
            print(f'\n----- EXPERIMENT {genres_acc[unique]} -----')
        else:
            print(f'\n-----  EXPERIMENT {unique} -----')
        f1_all = []
        acc_all = []
        prec_all = []
        rec_all = []
        all_best_coefs = []
        for seed in range(5):
            print(f'----- SEED {seed + 1} -----')
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            y_kfold_pred = []
            y_kfold_te = []
            new_X, new_y = _balance_for_bin(X, y, unique, seed=seed)
            for i, (train_index, test_index) in enumerate(kfold.split(new_X, new_y)):
                print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
                X_tr = new_X[train_index]
                X_te = new_X[test_index]
                y_tr = new_y[train_index]
                y_te = new_y[test_index]
                X_tr, X_te, feat_names = getattr(preprocessing.feature_extraction, f'extract_features_{feat}')(X_tr, X_te, y_tr)
                print("Training shape: ", X_tr.shape)
                print("Test shape: ", X_te.shape)
                cls = LinearSVC(random_state=42).fit(X_tr, y_tr)
                y_pred = cls.predict(X_te)
                coefs = np.array(cls.coef_[0])
                coefs_idx = (-coefs).argsort()[:3]
                best_coefs = [feat_names[idx] for idx in coefs_idx]
                all_best_coefs.extend([best_coef for best_coef in best_coefs if best_coef not in all_best_coefs])
                y_kfold_pred.extend(y_pred)
                y_kfold_te.extend(y_te)
            print(f'----- KFOLD RESULTS -----')
            f1 = f1_score(y_kfold_te, y_kfold_pred, average='binary')
            f1_all.append(f1)
            print(f'F1: {f1:.3f}')
            acc = accuracy_score(y_kfold_te, y_kfold_pred)
            acc_all.append(acc)
            print(f'Accuracy: {acc:.3f}')
            prec = precision_score(y_kfold_te, y_kfold_pred, average='binary')
            prec_all.append(prec)
            print(f'Precision: {prec:.3f}')
            rec = recall_score(y_kfold_te, y_kfold_pred, average='binary')
            rec_all.append(rec)
            print(f'Recall: {rec:.3f}')
        print(f'----- FINAL RESULTS -----')
        f1 = round(stat.mean(f1_all), ndigits=3)
        print(f'F1: {f1:.3f}')
        acc = round(stat.mean(acc_all), ndigits=3)
        print(f'Accuracy: {acc:.3f}')
        prec = round(stat.mean(prec_all), ndigits=3)
        print(f'Precision: {prec:.3f}')
        rec = round(stat.mean(rec_all), ndigits=3)
        print(f'Recall: {rec:.3f}')
        print('Best coefs:', all_best_coefs)
        rows.append([unique, f1, acc, prec, rec, all_best_coefs])
        if save:
            if not os.path.exists('output'):
                os.mkdir('output')
            with open(f'output/bin_{domain}_{feat}.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([domain, 'f1', 'accuracy', 'precision', 'recall', 'best_coefs'])
                writer.writerows(rows)


def round_of_kcv(domain, feat):
    macros = []
    micros = []
    for seed in range(10):
        dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=False,
                                as_dict=True, random_authors=5, seed=seed)
        print('\n')
        print('#samples:', len(dataset['lyrics']))
        print('#artists:', len(np.unique(dataset['artists'])))
        print('#genres:', len(np.unique(dataset['genres'])))

        ma, mi = kcv_classification(dataset, domain=domain, feat=feat)
        macros.append(ma)
        micros.append(mi)
    print('MACRO-F1 MEAN:', round(stat.mean(macros), ndigits=3))
    print('MICRO-F1 MEAN:', round(stat.mean(micros), ndigits=3))
