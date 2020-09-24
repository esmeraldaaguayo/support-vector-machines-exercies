% Diabetes dataset

load diabetes;

Xtrain = trainset;
Ytrain = labels_train;
Xtest = testset;
Ytest = labels_test;
% 
% join = [X,Y];
% 
% [~,ax] = plotmatrix(join);
% ax(1,1).YLabel.String='F1'; 
% ax(2,1).YLabel.String='F2'; 
% ax(3,1).YLabel.String='F3';
% ax(4,1).YLabel.String='F4'; 
% ax(5,1).YLabel.String='F5'; 
% ax(6,1).YLabel.String='F6'; 
% ax(7,1).YLabel.String='F7'; 
% ax(8,1).YLabel.String='F8'; 
% ax(9,1).YLabel.String='Classes'; 
% ax(9,1).XLabel.String='F1';
% ax(9,2).XLabel.String='F2';
% ax(9,3).XLabel.String='F3';
% ax(9,4).XLabel.String='F4';
% ax(9,5).XLabel.String='F5';
% ax(9,6).XLabel.String='F6';
% ax(9,7).XLabel.String='F7';
% ax(9,8).XLabel.String='F8';
% ax(9,9).XLabel.String='Classes';

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
plot(Xpoly,Ypoly);
plot(Xrbf,Yrbf);
legend('Linear LSSVM','Polynomial LSSVM','RBF kernel LSSVM');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Classifiers')
hold off
