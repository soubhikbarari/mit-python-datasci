# -*- coding: utf-8 -*-
"""
Application: Analyzing party affiliation, ideology, and coalitions
             in 1984 U.S. House of Representatives using roll-call 
             votes on 16 issues.
Author:      Soubhik Barari


"""
#-----------#
# Libraries #
#-----------#
import numpy      as np
import pandas     as pd
import matplotlib as mpl
import sklearn    as skl

import matplotlib.pyplot as plt

import sklearn.decomposition
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics

#---------------------------------------------------#
# Read in, re-code data, remove incomplete records, #
# summarise data                                    #
#---------------------------------------------------#

## read
df = pd.read_csv("./data/house-votes-84.csv")

## re-code
issues = df.columns[1:]
df = df.replace({
            "y" : 1,
            "n" : 0,
            "?" : None
        })
df = df.dropna()

## summarise
summary_df = df.groupby("party").sum()


## finalize
df = df.replace({ "democrat" : 1, "republican" : 0 })

#------------------------------------------#
# Q1: Can we predict party based on vote?  #
#     (classification problem)             #
#------------------------------------------#


## [A.] Use "off-the-shelf" Random Forest classifier

rf = skl.ensemble.RandomForestClassifier(n_estimators=100,
                                         max_features="auto")

X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(df[issues], 
                                                                        df["party"], 
                                                                        test_size=0.25, 
                                                                        random_state=42)

rf.fit(X_train, y_train)

y_test_pred = rf.predict(X_test)

print "Prediction accuracy:", sklearn.metrics.accuracy_score(y_test, y_test_pred)


## [B.] Create hyperparameter optimized Random Forest classifier

rf = skl.ensemble.RandomForestClassifier()
rf_hyperparam_options = {
            "n_estimators" : [100, 300, 500],
            "max_features" : [5, 10, 15]
        }

X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(df[issues], 
                                                                        df["party"], 
                                                                        test_size=0.25, 
                                                                        random_state=42)

grid_searcher = skl.model_selection.GridSearchCV(estimator=rf, 
                                                 param_grid=rf_hyperparam_options,
                                                 verbose=1)

grid_searcher.fit(X_train, y_train)

#---------------------------------------------------#
# Q2: How polarized was ideology in the '84 house?  #
#     (dimensionality reduction problem)            #
#---------------------------------------------------#

X = df[issues]
y = df["party"]

## Find linear combination of issues (along 2 dimensions) 
## that best explains variance in voting pattern
pca = sklearn.decomposition.PCA(n_components=2)

X_2d  = pca.fit(X).transform(X)
print "Variance explained by first 2 components: %s" % pca.explained_variance_ratio_

## Now let's map party affiliation onto voting along these dimensions!
## Note: if running in `Spyder`, must call all of these below to plot onto one figure.
plt.scatter(X_2d[np.argwhere(y == 0),0], X_2d[np.argwhere(y == 0),1], label="Rep.", c="r")
plt.scatter(X_2d[np.argwhere(y == 1),0], X_2d[np.argwhere(y == 1),1], label="Dem.", c="b")
plt.title("Ideology in the 1984 House of Rep.", fontweight="bold")
plt.xlabel("First issue dimension")
plt.ylabel("Second issue dimension")
plt.legend()
plt.savefig("imgs/house-84-ideology.pdf")

issue_dimensions_df = pd.DataFrame(pca.components_,columns=issues, index = ['PC-1','PC-2'])

#----------------------------------------------------------#
# Q3: Are there are any detectable ideological coalitions? #
#     (clustering problem)                                 #
#----------------------------------------------------------#

# TODO
