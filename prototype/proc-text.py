import re
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle

directory = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

# read tab separated dataset with training and testing instances
df_train = pd.read_table(f"{directory}/data/exercise_train.tsv")
df_test = pd.read_table(f"{directory}/data/exercise_test.tsv")

# set aside 10% of the training data for validation purposes
ds_train, ds_validate = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['pattern'].values)

# check column names
# df_train.columns
# images here
# df_train.iloc[0,5:9]
# iterate over rows
# for row in df_train.itertuples():
# print(row[6])

ctab = pd.crosstab(index=df_train['pattern'], columns='count')

# pd.crosstab(index=df_train['productType'],columns='count')
# pd.crosstab(index=df_train['gender'],columns='count')


x_train = df_train['name'].str.cat(df_train['description'], sep=';').values
# remove numbers from text
for i in range(len(x_train)):
    x_train[i] = re.sub(r"[0-9]+", "", x_train[i])

y_train = df_train['pattern']

# remove additional words assuming they are irrelevant to classification
stops = set(text.ENGLISH_STOP_WORDS)
stops.update(('dress', 'model', 'wears', 'fit', 'true', 'to', 'size', 'uk', 'us', 'tall', 'cm'))

# using simple term frequency with inverse document frequency normalisation
# using gradient descent classifier for scalability as well as flexibility
# using 'huber' loss since more suitable for one-vs-all classification
text_clf = Pipeline([('tfidf_vect', TfidfVectorizer(stop_words=stops)),
                     # ('std', StandardScaler(with_mean=False)),
                     # ('std', MaxAbsScaler()),
                     ('clf', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=1e-3,
                                           random_state=42))])

text_clf.fit(x_train, y_train)

y_pred = text_clf.predict(x_train)

np.mean(y_pred == y_train)

print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

# extract model coefficients and feature names (token) from the pipeline
coefs = text_clf.named_steps['clf'].coef_
tvec = np.array(text_clf.named_steps['tfidf_vect'].get_feature_names())

# check the model coefficient to get the most "important" features per class
# ranking would make more sense if coefficients were standardised
for i in range(coefs.shape[0]):
    top5 = np.argsort(-coefs[i])[:5]
    print(ctab.index[i], tvec[top5])

# check model sparsity
for i in range(coefs.shape[0]):
    print(ctab.index[i], "sparsity", np.sum(coefs[i] == 0) / coefs.shape[1])

# feature extraction and classifier parameters to possibly tune
parameters = {'tfidf_vect__max_df': (0.5, 0.75, 1.0),
              # 'tfidf_vect__max_features': (None, 5000, 10000, 50000),
              'tfidf_vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
              'clf__alpha': (1e-05, 1e-04, 1e-3),
              # 'clf__penalty': ('l2', 'elasticnet', 'l1'),
              'clf__l1_ratio': (0.01, 0.05, 0.1),
              # 'clf__n_iter': (5, 10, 50),
              }

# stratified CV to try and preserve class distribution
skf = StratifiedKFold(n_splits=8)

# grid search for best parameters combination
grid_search = GridSearchCV(text_clf, param_grid=parameters, n_jobs=-1, cv=skf, verbose=1)
grid_search.fit(x_train, y_train)

# check best scoring model
print("Best score:", grid_search.best_score_)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print(param_name, best_parameters[param_name])

text_clf = grid_search.best_estimator_

# increase number of folds to test final model
ccr = cross_val_score(text_clf, x_train, y_train, cv=16, n_jobs=-1)
np.mean(ccr)

# prediction on unseen data (I should have made a custom pre-processor for the pipeline!)
x_test = df_test['name'].str.cat(df_test['description'], sep=';').values
for i in range(len(x_test)):
    x_test[i] = re.sub(r"[0-9]+", "", x_test[i])

pred_test = text_clf.predict(x_test)

res = pd.Series(pred_test, index=df_test['productIdentifier'])
res.to_csv(f'{directory}/data/predictions_test_text.csv')

model_path = os.path.join(directory, 'saved_model', 'model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as model_file:
    pickle.dump(text_clf, model_file)

