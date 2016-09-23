function [error] = kernreg_xval_error(sigma, X, Y, part, distFunc)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(SIGMA, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KERNREG_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNREG_TEST

N = max(part);
sumz = zeros(N,1);
for i = 1:N
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

    testLabels = kernreg_test(sigma,train,Ytrain,test,distFunc);
    difference = testLabels-Ytest;
    difference = difference~=0;
    j = sum(difference);
    [m,n] = size(difference);
    sumz(i) = j/m;
end
error =(sum(sumz))/N
