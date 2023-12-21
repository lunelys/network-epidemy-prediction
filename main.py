# Author: Lunelys RUNESHAW <lunelys.runeshaw@etudiant.univ-rennes.fr>
# Master 2, Project for the UE Machine Learning in Biology, last edit 20/12/23
# Influenza outbreak event prediction via Twitter based on the article:
# "SimNest: Social Media Nested Epidemic Simulation via Online Semi-supervised Deep Learning" (2015, L. Zhao)
# Python 3.10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # another package to plot (but pretty)

# Packages for the PCA
from sklearn.decomposition import PCA  # sklearn is a machine learning library

# Packages for the PCoA/MDS (didn't work)
# First try
# from sklearn.manifold import MDS
# Second try
# from skbio import DistanceMatrix
# from skbio.stats.ordination import pcoa

from sklearn import preprocessing  # to normalize

# Packages for the decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics  # for the accuracy calculation
from sklearn import tree  # to plot the tree

# Packages for the random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV  # to iteratively test models to get a "best" model
from scipy.stats import randint
from sklearn.tree import plot_tree

# packages for the neural network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # (maybe already normalized with X_train_scaled?)

# pd.options.display.max_columns = 500 # If you want to display really ALL columns

# Preparation of the train set: -----------------------------------------
X_train = pd.read_csv("Inlufenza_set2.csv")
y_train = pd.read_csv("Inlufenza_set2_label.csv")
# initially the labels column is called "x", which is confusing. Here we rename it to y:
y_train.rename(columns={'x': 'y'}, inplace=True)
# Delete in X_train last columns (unlabeled) and first descriptive columns
X_train.drop(X_train.iloc[:, 527:], axis=1, inplace=True)
X_train.drop(X_train.iloc[:, :2], axis=1, inplace=True)
# Normalizing by using frequencies instead of the integer values
# so that the square root of the sum of the squares of the values equals one
# https://www.digitalocean.com/community/tutorials/normalize-data-in-python
d = preprocessing.normalize(X_train)  # normalize per row because a row is a week
X_train_scaled = pd.DataFrame(d, columns=X_train.columns)
# Joins X_train and y_train
train = pd.concat([X_train_scaled, y_train], sort=False, axis=1)

# Preparation of the test set: -----------------------------------------
X_test = pd.read_csv("Inlufenza_set1.csv")
y_test = pd.read_csv("Inlufenza_set1_label.csv")
# initially the labels column is called "x", which is confusing. We rename it to y here too:
y_test.rename(columns={'x': 'y'}, inplace=True)
# Again dropping the useless columns
X_test.drop(X_test.iloc[:, 527:], axis=1, inplace=True)
X_test.drop(X_test.iloc[:, :2], axis=1, inplace=True)
d = preprocessing.normalize(X_test)  # normalize per row because a row is a week
X_test_scaled = pd.DataFrame(d, columns=X_test.columns)
test = pd.concat([X_test, y_test], sort=False, axis=1)  # join on test set too

# # Check of how the test data looks
# print(X_test.shape)  # (23280, 525)
# print(X_test.head())
# #    flu  swine  stomach  symptoms  ...  complications  children  start  aja
# # 0    0      0        0         0  ...              0         0      0    0
# # 1    0      0        0         0  ...              0         0      0    0
# # 2    0      0        0         0  ...              0         0      0    0
# # 3    0      0        0         0  ...              0         0      0    0
# # 4    0      0        0         0  ...              0         0      0    0
# print(y_test.shape)  # (23280, 1)
#
# print(X_train.shape)  # (52560, 525)
# print(y_train.shape)  # (52560, 1)
# print(train.shape)  # (52560, 526)
# print(train)  # y is the labels
# #        flu  swine  stomach  symptoms  ...  children  start  aja  y
# # 0        0      0        0         0  ...         0      0    0  0
# # 1        0      0        0         0  ...         0      0    0  0
# # 2        0      0        0         0  ...         0      0    0  0
# # 3        0      0        0         0  ...         0      0    0  0
# # 4        0      0        0         0  ...         0      0    0  0
# #     ...    ...      ...       ...  ...       ...    ...  ... ..
# # 52555    1      1        1         1  ...         0      0    0  1
# # 52556    1      1        1         2  ...         0      0    0  1
# # 52557    0      0        0         0  ...         0      0    0  1
# # 52558    0      0        0         0  ...         0      0    0  1
# # 52559    0      0        0         0  ...         0      0    0  1
#
#
# # Random exploratory, to understand a bit better the data: ---------NOT FOR THE STUDY ------------
# print(train["flu"].unique())
# # which values in the flu columns are unique? For the flu:
# # [  0   1   3   4   2   5   6   8   9  10   7  24  11  28  18  15  12  13
# #   20  14  29  17  23  16  21  19  32  31  75  55  41  56  27  25  35  80
# #   30  22  26  44  33  47 152  39  38  42  37  36  43  34  52  45  46]
# # => we have a peak of 152 tweets with the flu keyword
# print(train["children"].unique())
# # [0 1 2 3 4]
#
# # Now, if we try to plot the frequency of the values in the flu column, for example
# train["flu"].plot.hist()
# plt.ylim(0, 200)  # to actually see the 152 tweets frequency (so low compared to 0-40 tweets)
# plt.show()
# # We can see that the HUGE majority is 0 to 40 tweets. The 152 peak has a VERY low frequency

# ------------------------------------------------------------------------------------------

# 1) PCA: look what it looks like =====================================================================================
pca_keywords = PCA(n_components=2)  # Let's try with only 2 components at the beginning
PC_keywords = pca_keywords.fit_transform(X_train_scaled)  # contains the values of the 2 PCs for ALL rows
PC_keywords_df = pd.DataFrame(data=PC_keywords, columns=['PC1', 'PC2'])  # we create a df from those values
print(pca_keywords.explained_variance_ratio_)  # [0.43246992 0.16201601]
# The first PC here keeps 43% of the information, and the second 16%. It's more than half the info summarized in 2 components!
PC_keywords_df['y'] = y_train

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="PC1", y="PC2",
    hue="y",  # the thing to predict
    palette=sns.color_palette("hls", 2),  # only 2 colors because Y is only 0 or 1
    data=PC_keywords_df,
    legend="full",
    alpha=0.3  # that's just the transparency of points, to better see when they are stacked
)
plt.savefig('PCA_train_dataset', bbox_inches='tight')
plt.show()  # fan shape
# So here the coloring is the label (which is a given). The points are scattered depending on their coordinates as PC1 and PC2
# We see that there are a few outliers that make the plot a bit ugly. Not much of a division can be spotted easily here.
# . Possibly not good then? (PS: it was because it was not scaled)
# To make a more centered version FOR THE NOT SCALED:
# plt.ylim(-100, 100)
# plt.xlim(-50, 400)
# plt.savefig('PCA_train_dataset', bbox_inches='tight')
# plt.show()

# PCA on the test set, no plotting. Useful for the decision tree later
pca_keywords_test = PCA(n_components=2)
PC_keywords_test = pca_keywords_test.fit_transform(X_test_scaled)  # contains the values of the 2 PCs for ALL rows
PC_keywords_test_df = pd.DataFrame(data=PC_keywords_test, columns=['PC1', 'PC2'])  # we create a df from those values
print(pca_keywords_test.explained_variance_ratio_)  # [0.85169887 0.1079266 ]
# first PC here keep 85% of the information, and the second 10%.
PC_keywords_test_df['y'] = y_test

# 1 bis) Actually, it's better to do a PCoA
# # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
# # And now we perform the PCoA (Principal Coordinate Analysis), because it takes into account the effectifs
# # Great explanations: https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/
# Euclidean distance: Bray-Curtis Vs Jaccard
# First try: doesn't work
# dm = DistanceMatrix(train, ids=sample_names)  # Load the pandas matrix into skbio format
# plt.style.use('ggplot')  # Set plot style
# pcoa_results = pcoa(dm)
# fig = pcoa_results.plot(df=groups, column='Cluster', cmap='Set1', s=50)  # groups and 'Cluster' are metadata
# # pcoa_results.samples[['PC1', 'PC2']]
# plt.show()

# Second try: doesn't work either
# embedding = MDS(n_components=2, normalized_stress='auto')
# X_transformed = embedding.fit_transform(X_train_scaled)
# print(X_transformed)


# # 2) do a decision tree ==========================================================================================
# We will have to use the test set here (Cross-validation necessary)!

clf = DecisionTreeClassifier()  # Here we create the Decision Tree classifier object
clf = clf.fit(X_train.values, y_train.values)  # ... and we train it.
y_pred = clf.predict(X_test.values)  # Now we predict the response for the test dataset (X_test)
# Note: why .values? It is just to trim the header. Otherwise we would have the full df, not good
print(metrics.accuracy_score(y_test.values, y_pred))  # 0.8780498281786941
# 87% of model accuracy, basically how often is the classifier correct: that's not bad for a method as simplistic as decision trees!!

# Let's plot the tree now!
tree.plot_tree(clf)
plt.savefig('unpruned_decision_tree', bbox_inches='tight')
# plt.show()
# Tree is unreadable, way too deep, which makes sense with our 500+ features! Overfitting issue

# So, let's prune it. We redo a new one with a few additional parameters:
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # we could try with the Gini criterion too
clf = clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
print(metrics.accuracy_score(y_test.values, y_pred))  # 0.9019759450171821 : 3% better after pruning!
tree.plot_tree(clf)
plt.savefig('pruned_decision_tree', dpi=300)
plt.show()
# Explanation: we limited the tree depth to 3 (max_depth=3), to avoid overfitting. The x[number] in the tree are the
# indexes of the features. So here for example X[0] is the flu keyword. So in the tree generated here, x[20] is the BIGGEST
# divider (because it's the first one). Let's check what is x[20]. Then you continue to divide along 2 more depth. Check to what
# corresponds x[19] and the others that appear in the tree (= the important keywords).
# Possibility: try with the Gini criterion. Try a bit more depth?
# print(X_train.columns[20])  # soon ... a bit weird.
# print(X_train.columns[19])  # immune.... makes a bit more sense but not great
# print(X_train.columns[147])  # gym
# print(X_train.columns[438])  # goin
# print(X_train.columns[235])  # etc
# print(X_train.columns[388])  # cancer
# We could have thought the most important divider would have been somthing like flu or something like that, but no
# -> to investigate, maybe the criterion used (entropy) is not good. PS: and a decision tree is too simple
feature_importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances.iloc[:50].plot.bar()  # and here we plot a simple bar chart based on the 50 most important features
plt.savefig('feature_importance_decision_tree', bbox_inches='tight')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy Decision Tree:", accuracy)  # 0.9019759450171821
print("Precision Decision Tree:", precision)  # 0.5
print("Recall Decision Tree:", recall)  # 0.011393514460999123

# # # => Let's do it with the dimension reduction results: (PS: no, shouldn't do it)
# # So now, the same, but with the PCA result (we can keep the previous for reference):
# clf_PCA = DecisionTreeClassifier()
# clf_PCA = clf_PCA.fit(PC_keywords_df[['PC1', 'PC2']].values, PC_keywords_df["y"].values)
# y_pred_PCA = clf_PCA.predict(PC_keywords_test_df[['PC1', 'PC2']].values)
# print(metrics.accuracy_score(PC_keywords_test_df['y'].values, y_pred_PCA))  # around 85%
# # tree.plot_tree(clf_PCA)  # plotting part
# # plt.show()
# # Tree is unreadable, but it is because it is unpruned! Now let's prune it.
# clf_PCA = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# clf_PCA = clf_PCA.fit(PC_keywords_df[['PC1', 'PC2']].values, PC_keywords_df["y"].values)
# y_pred_PCA = clf_PCA.predict(PC_keywords_test_df[['PC1', 'PC2']].values)
#
# # print(metrics.accuracy_score(PC_keywords_test_df['y'].values, y_pred_PCA))  # around 90% : it is 5% better!
# tree.plot_tree(clf_PCA)  # plotting part
# plt.savefig('pruned_PCA_decision_tree', dpi=300)  # to see it better, higher resolution than default
# plt.show()
# # same comment as earlier, but this time we have only 2 possibility of divider, as we have only 2 features with that PCA version:
# # PC1 (x[0]), and PC2 (x[1]).


# 3) Random Forest =======================================
# Training the model
rf = RandomForestClassifier()  # we keep the default parameters
# why .ravel()? https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
rf.fit(X_train, y_train.values.ravel())  # we train the random forest model

# Visualizing one of the tree of the trained random forest
tree_to_plot = rf.estimators_[0]  # Pick one tree from the forest, for example the first tree
plt.figure(figsize=(20, 10))
# we limit it to a depth of 4 to keep it readable, but it is way bigger;
# color code: red = majority of y = 0 (so not sick), blue = majority of y = 1 (sick)
plot_tree(tree_to_plot, feature_names=X_train.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=4)
plt.title("Decision Tree from Random Forest")
plt.savefig('example_first_tree_in_random_forest', bbox_inches='tight')
plt.show()

# Accuracy of the model?
y_pred = rf.predict(X_test)  # to check if the model is making accurate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest model accuracy:", accuracy)  # 0.9029209621993127
# It is a pretty good score, but we can try to improve it even more by fine-tuning the hyperparameters!

# Hyperparameter Tuning
# We will use RandomizedSearchCV from scikit-learn which randomly search parameters within a range per hyperparameter
param_dist = {'n_estimators': randint(50, 500),  # number of decision trees in the forest: not too much, but enough
              'max_depth': randint(1, 20)}  # same, should not be too low nor too high
rand_search = RandomizedSearchCV(rf,
                                 param_distributions=param_dist,
                                 n_iter=5,  # will train 5 models, among which will be best_rf
                                 cv=5)  # use random search to find the best hyperparameters
# Note: if you run it, there will be lots of warning; you can overlook them
rand_search.fit(X_train, y_train)  # fit the random search object to the data
best_rf = rand_search.best_estimator_  # create a variable for the best model
print('Best hyperparameters:',  rand_search.best_params_)  # 'max_depth': 13, 'n_estimators': 254 (change at each run)
# -> 254 trees would be optimal, with a forest depth of 13 levels

# Let's make a confusion matrix to check visually how is performing the model
y_pred = best_rf.predict(X_test)  # Generate predictions with the best model
cm = confusion_matrix(y_test, y_pred)  # Create the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.savefig('confusion_matrix_random_forest', bbox_inches='tight')
plt.show()  # For this specific run: 20997 true negative, 2259 false negative, 1 false positive, and 23 true positive

# Let's assess the performances of the best model (still random so will vary)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy best model RF:", accuracy)  # 0.9029209621993127
print("Precision best model RF:", precision)  # 0.9583333333333334
print("Recall best model RF:", recall)  # 0.010078878177037686
# That's not much better: probably because we repeat it only on 5 models (but it already takes quite some time to run)

# Create a series containing feature importances from the best model, using the model’s internal score to find the best way to split the data within each decision tree
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances.iloc[:50].plot.bar()  # and here we plot a simple bar chart based on the 10 most important features
plt.savefig('feature_importance_randomForest', bbox_inches='tight')
plt.show()  # For this specific run, I ended up with the 10 most significant features that follow:
# soon, gym, husband, vomiting, tomorrow, helps, dying, immune, etc, felt
# -> Not great but not too bad. Once again, it's probably because we repeat it only on 5 models to pick "the best"

# So now, let's redo a random forest, but directly with prof's proposed parameters: 500-1000 trees, 4 depth, 10 random picks
rf = RandomForestClassifier(n_estimators=1000, max_depth=4, max_features=10)  # 1rst run with 500 trees, 2nd with 1000
rf.fit(X_train, y_train.values.ravel())
y_pred = rf.predict(X_test)  # to check if the model is making accurate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest model accuracy:", accuracy)  # 500 trees: 0.9019759450171821; 1000 trees: 0.9019759450171821 (same)
# Let's make a confusion matrix to check visually how is performing the model
cm = confusion_matrix(y_test, y_pred)  # Create the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.tight_layout()
plt.savefig('confusion_matrix_random_forest_prof_1000_trees', bbox_inches='tight')
plt.show()  # For 500 trees: 20998 true negative, 2282 false negative, 0 false positive, and 0 true positive
# For 1000 trees: Same as 500 trees
# 0 true positive is worrying, we are supposed to have a few...

feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances.iloc[:50].plot.bar()  # and here we plot a simple bar chart based on the 50 most important features
# plt.show()  # see saved plot _prof_500_trees; kind of the same as the best_rf one actually
plt.savefig('feature_importance_randomForest_prof_1000_trees', bbox_inches='tight')  # 1000 trees; same as 500 trees.

# 4) Neural Network =======================================
# official documentation: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# (awesome resource to understand better the parameters below!)
# Note: with MLP, we REALLY need to normalize the data prior!!
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  # normalize train data
X_test = scaler.transform(X_test)  # apply same transformation to test data

# Now the data is correctly normalized, we can create the Neural Network
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,  # sgd gives similar results
                    hidden_layer_sizes=(5, 2), random_state=1, max_iter=100)
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))  # 0.963413
print("Test set score: %f" % mlp.score(X_test, y_test))  # 0.900773

# # Let's make a confusion matrix to check visually how is performing the model
y_pred = mlp.predict(X_test)
cm = confusion_matrix(y_test, y_pred)  # Create the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.tight_layout()
plt.savefig('confusion_matrix_mlp', bbox_inches='tight')
plt.show()  # 20673 true negative, 297 true positive, 1985 false negative and 325 false positive
# seems okay: we have true positive at least here

print(mlp.coefs_)  # coeffs for each of the 525 features
# [array([[-0.19157328, -0.00211649, -0.46248789, -0.23465773, -0.64569155],
#        [-0.37323735,  0.10880213,  0.24427682, -0.28512251,  0.33253146],
#        [-0.27755766,  0.15253441,  0.34339813, -0.11638337,  0.34642198],
#        ...,
#        [-0.14757556,  0.12777217, -0.02730383,  0.10051895, -0.15458126],
#        [ 0.0299219 , -0.19288348,  0.03139683, -0.15134654,  0.020755  ],
#        [-0.03234709, -0.03727873,  0.10148126,  0.047694  , -0.00996662]]), array([[-0.40304056, -0.9803531 ],
#        [-0.63296679,  0.94728691],
#        [-1.27169051,  0.44247506],
#        [ 0.7663307 ,  0.33860729],
#        [-0.63260841,  1.049264  ]]), array([[ 1.10660309],
#        [-1.50340976]])]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy MLP:", accuracy)  # 0.9007731958762887
print("Precision MLP:", precision)  # 0.477491961414791
print("Recall MLP:", recall)  # 0.13014899211218228

# No feature importance for mlp? Explanation: https://datascience.stackexchange.com/questions/44700/how-do-i-get-the-feature-importace-for-a-mlpclassifier
# interesting: https://www.datacamp.com/tutorial/layers-neurons-artificial-neural-networks