# imbalance_classification_test

Testing the classification of imbalanced data based on the paper "Exploratory Undersampling for Class-Imbalance Learning"

Input: sampling methods (ensemble, over-sampling, under-sampling, etc.) and classifier (AdaBoost and Random Forest)
Output: [sampling time, neg_count, pos_count, auc, f-score, g-mean]

Default testing settings and parameters:

Each data is tested with 10-fold stratified cross validation

Each method is tested 5 times

In each test time, the classification method is repeated 10 times 
(ensemble->n_estimators = 10, non-ensemble-> repeat = 10)

For non-ensemble methods, the number of classification iterations is 40

The number of subsets for ensemble methods is 4 


