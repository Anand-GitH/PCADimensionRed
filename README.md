# PCADimensionRed
PCA a powerful dimension reductionality tool

PCA Algorithm:
-Find mean of each predictor and subtract each value with the mean – centering the
values of all predictors
-Find the variance and covariance of all predictors to form the covariance matrix
-Use eigen decomposition of the covariance matrix to find the eigen values and eigen
vectors
-Sort the eigen values decreasing order
-Each eigen value represents each principal component and we can calculate amount of
variance each principal component shows in data.

Dataset: Pendigits – features of handwritten digits from 0 to 9
Classification using k-NN on raw data and PCA dimension reduced data
10992 observations and 16 predictors

Conclusion:

Test accuracy raw data is 99% and principal components data is 98% which is almost same and
with 7 principal components which is reduced data from 16 predictors to just 7 predictors and accuracy
is almost same. So PCA is very powerful tool for dimension reduction with which we can analyze how
data is distributed and reduce data dimension for building the simpler models with reduced data.
