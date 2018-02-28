This is my code for an in-class Kaggle Competetion. The competition was modified form a loan-delay-detection competition by erasing the column names and adding random noise as new features. 

For preprocessing, I divide the features into categorical and numerical and implement the null values with either most frequent values (for categorical features) or mean values (for numerical features). I then dummy encoded the categorical features and standardize the the numerical features.

For feature engineering, I emplement backward elemination on a fine tuning random forest model. PCA was not quite useful in this case.

My final best model is a two-level stacking model. I used Extra Trees, AdaBoost, Gradient Boost and Random Forest as first level classifier, and used another Random Forest as second level classifier.
