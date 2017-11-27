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

## Outlier Detection Experiments
#####Examples
tests/outlier_comparison_test_<dataset>.py

Each example is basically a python script, modifying several variables is needed to custom an outlier detection experiment.

#####Variables
log_file: target path of log recorded

noise_true_ratios: a list of ratios of _noise / true minority_



