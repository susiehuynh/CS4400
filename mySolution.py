#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from os.path import join

# 1. read data
# ltable = pd.read_csv("ltable.csv")
# rtable = pd.read_csv( "rtable.csv")
# train = pd.read_csv("train.csv")

# 1. read data
ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))

# In[3]:


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


def block_by_brand(ltable, rtable):
    # ensure brand is str
    ltable['brand'] = ltable['brand'].astype(str)
    rtable['brand'] = rtable['brand'].astype(str)

    # get all brands
    brands_l = set(ltable["brand"].values)
    brands_r = set(rtable["brand"].values)
    brands = brands_l.union(brands_r)

    # map each brand to left ids and right ids
    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}
    for i, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        brand2ids_r[x["brand"].lower()].append(x["id"])

    # put id pairs that share the same brand in candidate set
    candset = []
    for brd in brands:
#         print(brd)
        # if brd != '5.01E+11':
        l_ids = brand2ids_l[brd]
        r_ids = brand2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

# blocking to reduce the number of pairs to be compared
candset = block_by_brand(ltable, rtable)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)


# 3. Feature engineering
import Levenshtein as lev

def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

##### DROP NA IF NEED BE
# candset_df = candset_df.dropna(axis=0).reset_index(drop=True)
candset_features = feature_engineering(candset_df)



# 4. Model training and prediction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt

#split the training data
X_train, X_test, y_train, y_test = train_test_split(training_features, training_label, test_size=0.10)
# print("X_train.shape = ", X_train.shape)
# print("X_test.shape = ", X_test.shape)

# rf = RandomForestClassifier(class_weight="balanced", random_state=0)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(common_features)

#######################
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 5)
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)

#plot
# print(f"Classification Report for Random Forest\n\n")
# print(classification_report(y_test, y_pred))
# plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()

# ######################################################################### KNN

# from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)

# # predict
# y_pred = classifier.predict(X_test)

# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20, scoring="f1")
# print("KNN F1 Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ######################################################################### SVM

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)

# # predict
# y_pred = classifier.predict(X_test)

# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20, scoring="f1")
# print("SVM F1 Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# ######################################################################### NB Gaussian

# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# # predict
# y_pred = classifier.predict(X_test)

# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20, scoring="f1")
# print("NB Gaussian F1 Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# ######################################################################### NB Bernoulli

# from sklearn.naive_bayes import BernoulliNB
# classifier = BernoulliNB()
# classifier.fit(X_train, y_train)

# # predict
# y_pred = classifier.predict(X_test)

# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20, scoring="f1")
# print("NB Bernoulli F1 Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# #########################################################################

#applying cross_val_score to get best parameter
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20, scoring="f1")
# print("F1 Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [5, 10, 15, 20, 30], 'criterion': ['entropy', 'gini']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = 20,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
# print("Best F1 Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)

a = [i for i in best_parameters.values()]
best_classifier = RandomForestClassifier(criterion= a[0], n_estimators= a[1], random_state = 5)
best_classifier.fit(X_train, y_train)
# predict
y_pred = best_classifier.predict(X_test)



# 5. output
y_predicted_candset = classifier.predict(candset_features)
matching_pairs = candset_df.loc[y_predicted_candset == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)

