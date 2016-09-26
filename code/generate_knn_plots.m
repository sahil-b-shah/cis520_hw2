%% Script/instructions on how to submit plots/answers for question 2.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data: this loads X, Xnoisy, and Y.
load('/data/breast-cancer-data-fixed.mat');

%% 2.1
answers{1} = 'For regular data, the difference in error across N is negligible, although minimums tend to occur at N=4. For the noisy data, we see that the N-fold error tends to steadily decrease with increasing N (minimums most often at N=16), while the test data error stays roughly the same. However, even the difference in error in noisy data across N is small, but can be significant. It is noteworthy that the magnitude of error for noisy data is nearly 6 times the error of the regular data for both N-fold and test data. Standard deviations remain roughly constant.';
distFunc='l2';
xdata = X;
Npart = [2 4 8 16];
nfold_error = zeros(100,4,2);
true_error = zeros(100,4,2);
for z = 1:2
    for j = 1:4
        for i = 1:100
            [m,n] = size(xdata);
            order = randperm(m);
            ordertrain = order <401;
            ordertest = order >400;
            number = [1:m];

            vtest = number.*ordertest;
            vtest = vtest(vtest~=0);
            test = xdata(vtest,:);
            Ytest = Y(vtest,:);

            vtrain = number.*ordertrain;
            vtrain = vtrain(vtrain~=0);
            train = xdata(vtrain,:);
            Ytrain = Y(vtrain,:);

            part = make_xval_partition(400,Npart(j));
            if j<5
                nfold_error(i,j,z) = knn_xval_error(1,train,Ytrain,part,distFunc);

                testLabel = knn_test(1,train,Ytrain,test,distFunc);
                testLabel = round(testLabel);
                error = Ytest-testLabel;
                error = error~=0;
                r = sum(error);
                [m,n] = size(error);
                true_error(i,j,z) = r/m;
            end
        end
    end
    xdata = X_noisy;
end

nfold_errs = nfold_error(:,:,1);
size(nfold_error);
y = mean(nfold_errs);
e= std(nfold_errs);
x = Npart;
errorbar(x,y,e);
hold on;
test_error = true_error(:,:,1);
y = mean(test_error);
e = std(test_error);
errorbar(x,y,e,'r');
xlabel('N-folds');
ylabel('Average Error');
title('Average Cross Validation Error vs. N-fold Regular');
legend('N-fold error','Test error');
%print('-djpeg', 'C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\plot_2.1.jpg');
hold off;

figure(2);
nfold_errs = nfold_error(:,:,2);
y = mean(nfold_errs);
e= std(nfold_errs);
x = [2 4 8 16];
errorbar(x,y,e);
hold on;
test_error = true_error(:,:,2);
y = mean(test_error);
e = std(test_error);
errorbar(x,y,e,'r');
xlabel('N-folds');
ylabel('Average Error');
title('Average Cross Validation Error vs. N-fold Noisy');
legend('N-fold error','Test error');
%print('-djpeg', 'C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\plot_2.1-noisy.jpg');
hold off; 
% Plotting with error bars: first, arrange your data in a matrix as
% follows:
%
%  nfold_errs(i,j) = nfold error with n=j of i'th repeat
%  
% Then we want to plot the mean with error bars of standard deviation as
% folows: y = mean(nfold_errs), e = std(nfold_errs), x = [2 4 8 16].
% 
% >> errorbar(x, y, e);
%
% Along with nfold_errs, also plot errorbar for test error. This will 
% serve as measure of performance for different nfold-crossvalidation.
%
% To add labels to the graph, use xlabel('X axis label') and ylabel
% commands. To add a title, using the title('My title') command.
% See the class Matlab tutorial wiki for more plotting help.
% 
% Once your plot is ready, save your plot to a jpg by selecting the figure
% window and running the command:
%
% >> print -djpg plot_2.1-noisy.jpg % (for noisy version of data)
% >> print -djpg plot_2.1.jpg  % (for regular version of data)
%
% YOU MUST SAVE YOUR PLOTS TO THESE EXACT FILES.

%% 2.2
answers{2} = 'The best values appear to be {K=1,sigma=1} as these most often yield the lowest average error for both noisy and regular data. K data has a local minimum at K=8, while sigma data monotonically increases in error. The best average error is about 0.0425 and 0.25 for regular and noisy data respectively';
% distFunc = 'l2';
% k = [1,2,3,5,8,13,21,34];
% sigma = [1,2,3,4,5,6,7,8,9,10,11,12];
% xdata = X;
% for z = 1:2
%     for j = 1:12
%         for i = 1:100
%             [m,n] = size(xdata);
%             order = randperm(m);
%             ordertrain = order <401;
%             ordertest = order >400;
%             number = [1:m];
% 
%             vtest = number.*ordertest;
%             vtest = vtest(vtest~=0);
%             test = xdata(vtest,:);
%             Ytest = Y(vtest,:);
% 
%             vtrain = number.*ordertrain;
%             vtrain = vtrain(vtrain~=0);
%             train = xdata(vtrain,:);
%             Ytrain = Y(vtrain,:);
% 
%             part = make_xval_partition(400,10);
%             if j<9
% %                 nfold_error(i,j,z) = knn_xval_error(k(j),train,Ytrain,part,distFunc);
% %                 testLabel = knn_test(k(j),train,Ytrain,test,distFunc);
% %                 testLabel = round(testLabel);
% %                 error = Ytest-testLabel;
% %                 error = error~=0;
% %                 r = sum(error);
% %                 [m,n] = size(error);
% %                 true_error(i,j,z) = r/m;
%             end
%             
%             nfold_error2(i,j,z) = kernreg_xval_error(sigma(j), train, Ytrain,part,distFunc);
%             testLabel2 = kernreg_test(sigma(j),train,Ytrain,test,distFunc);
%             testLabel2 = round(testLabel2);
%             error2 = Ytest-testLabel2;
%             error2 = error2~=0;
%             r2 = sum(error2);
%             [o,p] = size(error2);
%             true_error2(i,j,z) = r2/o;
%         end
%     end
%     xdata = X_noisy;
% end

% figure(3);
% nfold_errs = nfold_error(:,:,1);
% y = mean(nfold_errs);
% e= std(nfold_errs);
% x = k;
% errorbar(x,y,e);
% hold on;
% test_error = true_error(:,:,1);
% y = mean(test_error);
% e = std(test_error);
% errorbar(x,y,e,'r');
% xlabel('K');
% ylabel('Average Error');
% title('Average Cross Validation Error vs. K Regular');
% legend('N-fold error','Test error');
% print('-djpeg', 'C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\plot_2.2-k.jpg');
% hold off;
% 
% figure(4);
% nfold_errs = nfold_error(:,:,2);
% y = mean(nfold_errs);
% e= std(nfold_errs);
% x = k;
% errorbar(x,y,e);
% hold on;
% test_error = true_error(:,:,2);
% y = mean(test_error);
% e = std(test_error);
% errorbar(x,y,e,'r');
% xlabel('K');
% ylabel('Average Error');
% title('Average Cross Validation Error vs. K Noisy');
% legend('N-fold error','Test error');
% print('-djpeg', 'C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\plot_2.2-k-noisy.jpg');
% hold off;

% figure(5);
% nfold_errs = nfold_error2(:,:,1);
% y = mean(nfold_errs);
% e= std(nfold_errs);
% x = sigma;
% errorbar(x,y,e);
% hold on;
% test_error = true_error2(:,:,1);
% y = mean(test_error);
% e = std(test_error);
% errorbar(x,y,e,'r');
% xlabel('Sigma');
% ylabel('Average Error');
% title('Average Cross Validation Error vs. Sigma Regular');
% legend('N-fold error','Test error');
% print('-djpeg','C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\plot_2.2-sigma.jpg');
% hold off;
% 
% figure(6);
% nfold_errs = nfold_error2(:,:,2);
% y = mean(nfold_errs);
% e= std(nfold_errs);
% x = sigma;
% errorbar(x,y,e);
% hold on;
% test_error = true_error2(:,:,2);
% y = mean(test_error);
% e = std(test_error);
% errorbar(x,y,e,'r');
% xlabel('Sigma');
% ylabel('Average Error');
% title('Average Cross Validation Error vs. Sigma Noisy');
% legend('N-fold error','Test error');
% print('-djpeg','C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\plot_2.2-sigma-noisy.jpg');
% hold off;

% Save your plots as follows:
%
%  noisy data, k-nn error vs. K --> plot_2.2-k-noisy.jpg
%  noisy data, kernreg error vs. sigma --> plot_2.2-sigma-noisy.jpg
%  regular data, k-nn error vs. K --> plot_2.2-k.jpg
%  regular data, kernreg error vs. sigma --> plot_2.2-sigma.jpg

%% Finishing up - make sure to run this before you submit.
save('C:\Users\Akshay Chandrasekhar\Downloads\hw2_kit\hw2_kit\problem_2_answers.mat', 'answers');
