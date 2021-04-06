import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


npzfile = np.load("./preprocessing_outputs/train_data.npz")
X = npzfile["data"]

npzfile = np.load("./preprocessing_outputs/test_data.npz")
test = npzfile["data"]

y = pd.read_csv('./preprocessing_outputs/train_labels.csv', index_col='building_id')
y = y.to_numpy().ravel()

print('X:', X.shape)
print('y:', y.shape)
print('test:', test.shape)

"""
param = {'n_estimators': [ 500, 1000], 'min_samples_split':[20, 50, 500]}

clf = RandomForestClassifier()

gd_sr = GridSearchCV(estimator=clf,
                     param_grid=param,
                     scoring='f1_micro',
                     cv=5,
                     n_jobs=-1)

gd_sr.fit(X, y)

print(gd_sr.best_score_)  # 50
print(gd_sr.best_params_) # 1000
"""


fin_clf = RandomForestClassifier(min_samples_split=50, n_estimators=1000)
fin_clf.fit(X, y)

y_pred = fin_clf.predict(X)
score  = f1_score(y, y_pred, average='micro')
print(score)



test    = np.nan_to_num(test)
results = fin_clf.predict(test)


def save_submission(results: pd.DataFrame, name: str, sample_submission_path: str) -> None:
    submission_format = pd.read_csv(sample_submission_path, index_col='building_id')
    submission = pd.DataFrame(results,
                              columns=submission_format.columns, index=submission_format.index)
    submission.to_csv(name)


save_submission(results, "submission_v3.csv", "../data/csv/submission_format.csv")
