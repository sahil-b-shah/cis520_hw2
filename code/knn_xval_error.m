function [error] = knn_xval_error(K, X, Y, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(K, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KNN_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KNN_TEST
z = max(part)
sumz = zeros(z,1)
for i = 1:z
    parttest = part==i;
    parttrain = part~=i;
    [m,n] = size(X);
    number = [1:m];

    vtest = number.*parttest;
    vtest = vtest(vtest~=0);
    test = X(vtest,:);
    Ytest = Y(vtest,:);

    vtrain = number.*parttrain;
    vtrain = vtrain(vtrain~=0);
    train = X(vtrain,:);
    Ytrain = Y(vtrain,:);

    testLabels = knn_test(K,train,Ytrain,test,distFunc);
    difference = testLabels-Ytest;
    difference = difference~=0;
    j = sum(difference);
    [m,n] = size(difference);
    sumz(i) = j/m;
end
sumz;
error =(sum(sumz))/z
%testLabels = knn_test(K,X,Y,test0,'l2');
