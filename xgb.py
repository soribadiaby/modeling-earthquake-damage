import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score 
from sklearn.model_selection import KFold


npzfile = np.load("./preprocessing_outputs/train_data.npz")
X = npzfile["data"]

npzfile = np.load("./preprocessing_outputs/test_data.npz")
test = npzfile["data"]

y = pd.read_csv('./preprocessing_outputs/train_labels.csv', index_col='building_id')
y = y.to_numpy().ravel()
y = y-1


def threshold_arr(array):
    # Get major confidence-scored predicted value.
    new_arr = []
    for ix, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val))))
        k[loc]=1
        new_arr.append(k)
        
    return np.array(new_arr)

SEED = 1881

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for ix, (train_index, test_index) in enumerate(kf.split(X)):
    lgb_params = {
        "objective" : "multiclass",
        "num_class":3,
        "metric" : "multi_error",
        "boosting": 'gbdt',
        "max_depth" : -1,
        "num_leaves" : 30,
        "learning_rate" : 0.1,
        "feature_fraction" : 0.5,
        "min_sum_hessian_in_leaf" : 0.1,
        "max_bin":8192,
        "verbosity" : 1,
        "num_threads":6,
        "seed": SEED
    }

    x_train, x_val, y_train, y_val= X[train_index], X[test_index], y[train_index], y[test_index]

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data   = lgb.Dataset(x_val, label=y_val)

    lgb_clf = lgb.train(lgb_params,
                        train_data,
                        20000,
                        valid_sets = [val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000)

    y_pred = lgb_clf.predict(x_val)
    print("F1-MICRO SCORE: ", f1_score(np.array(pd.get_dummies(y_val)), threshold_arr(y_pred), average='micro'))
    lgb_clf.save_model(f'models/model{ix}.txt')


# Load all LightGB Models and concatenate.
models = []
for i in range(5):
    model = lgb.Booster(model_file=f'models/model{i}.txt')
    y_pred = model.predict(X)
    score  = f1_score(np.array(pd.get_dummies(y)), threshold_arr(y_pred), average='micro')
    print("F1-MICRO SCORE: ", score)
    models.append(model)


def ensemble(models, x):
    # Ensemble K-Fold CV models with adding all confidence score by class.
    y_preds = []
    
    for model in models:
        y_pred = model.predict(x)
        y_preds.append(y_pred)
        
    init_y_pred = y_preds[0]
    for ypred in y_preds[1:]:
        init_y_pred += ypred
        
    y_pred = threshold_arr(init_y_pred)
    
    return y_pred


test = np.nan_to_num(test)
y_pred = ensemble(models, test)
y_pred = y_pred.argmax(axis=1)+1

def save_submission(results: pd.DataFrame, name: str, sample_submission_path: str) -> None:
    submission_format = pd.read_csv(sample_submission_path, index_col='building_id')
    submission = pd.DataFrame(results,
                              columns=submission_format.columns, index=submission_format.index)
    submission.to_csv(name)


save_submission(y_pred, "submission_xgb.csv", "../data/csv/submission_format.csv")