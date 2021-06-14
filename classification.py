from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from preprocessing.feature_extraction import extract_features_authorship


def kcv_classification(dataset, domain='authorship'):
    assert domain in ['authorship'], 'Only possible classification is with authorship atm'
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_all_pred = []
    y_all_te = []
    X = dataset['lyrics']
    y = dataset['artists']
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):

        print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
        X_tr = X[train_index]
        X_te = X[test_index]
        y_tr = y[train_index]
        y_te = y[test_index]
        X_tr, X_te = extract_features_authorship(X_tr, X_te, y_tr)
        print("Training shape: ", X_tr.shape)
        print("Test shape: ", X_te.shape)
        print('CLASSIFICATION')
        cls = KNeighborsClassifier(weights='distance', n_jobs=4)
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
