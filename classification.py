import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import preprocessing.feature_extraction
from preprocessing.preprocessing_classification import genres_acc
import os
import csv


def _balance_for_AV(X, y, unique):
    y = (y == unique).astype(int)  # positive_class = 1; negative_class = 0
    n_pos = np.count_nonzero(y == 1)
    if (n_pos * 2) >= (len(y) - n_pos):
        n_neg = len(y) - n_pos
        print(f'#pos:{n_pos} #neg:{n_neg}')
        return X, y
    else:
        n_neg = n_pos
        print(f'#pos:{n_pos} #neg:{n_neg}')
        np.random.seed(42)
        to_delete = np.random.choice(np.where(y == 0)[0], len(y)-(n_pos + n_neg), replace=False)
        new_X = np.delete(X, to_delete)
        new_y = np.delete(y, to_delete)
        return new_X, new_y


def kcv_classification(dataset, domain='artists', feat='base'):
    domains = ['artists', 'genres']
    feats = ['base', 'charngrams', 'wordngrams', 'prosody', 'all']
    assert domain in domains, f'Only possible domain are {domains}'
    assert feat in feats, f'Only possible features are {feats}'
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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


def AV_classification(dataset, domain='artists', feat='base', save=False):
    domains = ['artists', 'genres']
    feats = ['base', 'charngrams', 'wordngrams', 'prosody', 'all']
    assert domain in domains, f'Only possible domain are {domains}'
    assert feat in feats, f'Only possible features are {feats}'
    X = dataset['lyrics']
    y = dataset[domain]
    uniques = np.unique(y)
    rows = []
    for unique in uniques:
        if domain == 'genres':
            print(f'\n----- EXPERIMENT {genres_acc[unique]} -----')
        else:
            print(f'\n-----  EXPERIMENT {unique} -----')
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_all_pred = []
        y_all_te = []
        all_best_coefs = []
        new_X, new_y = _balance_for_AV(X, y, unique)
        for i, (train_index, test_index) in enumerate(kfold.split(new_X, new_y)):
            print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
            X_tr = new_X[train_index]
            X_te = new_X[test_index]
            y_tr = new_y[train_index]
            y_te = new_y[test_index]
            X_tr, X_te, feat_names = getattr(preprocessing.feature_extraction, f'extract_features_{feat}')(X_tr, X_te, y_tr)
            print("Training shape: ", X_tr.shape)
            print("Test shape: ", X_te.shape)
            print(f'----- CLASSIFICATION -----')
            cls = LinearSVC(random_state=42).fit(X_tr, y_tr)
            y_pred = cls.predict(X_te)
            f1 = f1_score(y_te, y_pred, average='binary')
            print(f'F1: {f1:.3f}')
            print(f'----- COEFFICIENTS -----')
            coefs = np.array(cls.coef_[0])
            coefs_idx = (-coefs).argsort()[:5]
            best_coefs = []
            for idx in coefs_idx:
                best_coefs.append((feat_names[idx], coefs[idx]))
                print(feat_names[idx], coefs[idx])
            all_best_coefs.append(best_coefs)
            y_all_pred.extend(y_pred)
            y_all_te.extend(y_te)
        print(f'----- FINAL RESULTS -----')
        f1_all = f1_score(y_all_te, y_all_pred, average='binary')
        acc_all = accuracy_score(y_all_te, y_all_pred)
        pres_all = precision_score(y_all_te, y_all_pred, average='binary')
        rec_all = recall_score(y_all_te, y_all_pred, average='binary')
        print(f'F1: {f1_all:.3f}')
        print(f'Accuracy: {acc_all:.3f}')
        print(f'Precision: {pres_all:.3f}')
        print(f'Recall: {rec_all:.3f}')
        rows.append([unique, f1_all, acc_all, pres_all, rec_all, all_best_coefs])
    if save:
        if not os.path.exists('output'):
            os.mkdir('output')
        with open(f'output/AV_{domain}_{feat}.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([domain, 'f1', 'accuracy', 'precision', 'recall', 'best_coefs'])
            writer.writerows(rows)



