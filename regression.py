from preprocessing import fetch_dataset
import numpy as np
from sklearn.model_selection import KFold
import preprocessing.feature_extraction
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize


def regression(domain, feat):
    dataset = fetch_dataset(pickle_path='data/dataset.pickle', lyrics_path='data/results_genre.csv', force=False,
                            as_dict=True, random_authors=50)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_all_pred = []
    y_all_te = []
    X = dataset['lyrics']
    y = dataset[domain]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
        X_tr = X[train_index]
        X_te = X[test_index]
        y_tr = y[train_index]
        y_te = y[test_index]
        X_tr, X_te, feat_names = getattr(preprocessing.feature_extraction, f'extract_features_{feat}')(X_tr, X_te, y_tr)
        print("Training shape: ", X_tr.shape)
        print("Test shape: ", X_te.shape)
        print('REGRESSION')
        #reg = LinearRegression().fit(X_tr, y_tr)
        #reg = DecisionTreeRegressor().fit(X_tr, y_tr)
        reg = SVR(kernel='linear').fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        print(y_te)
        print(y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        mse = mean_squared_error(y_te, y_pred)
        print(f'Mean Absolute Error: {mae:.3f}')
        print(f'Mean Squared Error: {mse:.3f}')
        y_all_pred.extend(y_pred)
        y_all_te.extend(y_te)

    print(f'----- FINAL RESULTS -----')
    mae_all = mean_absolute_error(y_all_te, y_all_pred)
    mse_all = mean_squared_error(y_all_te, y_all_pred)
    print(f'MAE: {mae_all:.3f}')
    print(f'MSE: {mse_all:.3f}')


