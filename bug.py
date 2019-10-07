import joblib
import numpy as np

rf = joblib.load('faulty_forest.pkl')
X = joblib.load('faulty_X.pkl')
y = joblib.load('faulty_y.pkl')
sample_weight = joblib.load('faulty_sample_weight.pkl')

rf.fit(X, y, sample_weight=sample_weight)

print('Feature importances of the forest', rf.feature_importances_)

faulty_trees = [t for t in rf.estimators_ if np.any(np.isnan(t.feature_importances_))]

print('Number of trees with NaN importances:', len(faulty_trees))
print('Feature importances of the faulty tree',
      faulty_trees[0].feature_importances_)

print('Potential cause is impurity 103 which is already: ',
      faulty_trees[0].tree_.impurity[103])

print('Interestingly, predicting still works:',
      rf.predict(X[:3, :]))
