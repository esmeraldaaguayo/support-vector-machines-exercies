% Wisconsin Breast Cancer dataset

load breast;

Xtrain = trainset;
Ytrain = labels_train;
Xtest = testset;
Ytest = labels_test;

% mal =0;
% neg =0;
% for i=1:size(Y,1)
%     if Y(i) == 1
%         mal = mal + 1;
%     else
%         neg = neg + 1;
%     end
% end
% 
% figure;
% pie([neg, mal]);
% hold off;

% LINEAR
% parameter tuning
[gamlin]=tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'lin_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
type='c'; 
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamlin,[],'lin_kernel'}); 
% figure; 
% plotlssvm({Xtrain,Ytrain,type,gamlin,[],'lin_kernel','preprocess'},{alpha,b});
[Yht, Ztlin] = simlssvm({Xtrain,Ytrain,type,gamlin,[],'lin_kernel'}, {alpha,b}, Xtest);
err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
[Xlin,Ylin,Tlin,AUClin] = perfcurve(Ytest, Ztlin, 1);


% POLYNOMIAL
[gampoly, last] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'poly_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
type='c'; 
t = last(1);
degree =last(2);
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gampoly,[t; degree],'poly_kernel'});
% figure; 
% plotlssvm({Xtrain,Ytrain,type,gampoly,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Ztpoly] = simlssvm({Xtrain,Ytrain,type,gampoly,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);
err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
[Xpoly,Ypoly,Tpoly,AUCpoly] = perfcurve(Ytest, Ztpoly, 1);


% RBF_KERNEL
[gamrbf, sig2] =tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
type='c'; 
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamrbf,sig2,'RBF_kernel'});
% figure;
% plotlssvm({Xtrain,Ytrain,type,gamrbf,sig2,'RBF_kernel','preprocess'},{alpha,b});
[Yht, Ztrbf] = simlssvm({Xtrain,Ytrain,type,gamrbf,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
[Xrbf,Yrbf,Trbf,AUCrbf] = perfcurve(Ytest, Ztrbf, 1);

figure;
plot(Xlin,Ylin);
hold on;
plot(Xrbf,Yrbf);
plot(Xpoly,Ypoly);
legend('Linear LSSVM','Polynomial LSSVM','RBF kernel LSSVM');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Classifiers')
hold off

