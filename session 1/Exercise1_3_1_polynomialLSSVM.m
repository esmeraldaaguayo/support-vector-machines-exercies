load iris

%
% Train the LS-SVM classifier using polynomial kernel
%
type='c'; 
gam = 1; 
t = 1; 
% degree = 11;
degreeList = [1,2,3,4,5,6,7,8,9,10,11,12,13];

errList=[];
for degree=degreeList
    fprintf('Polynomial kernel of degree %d ', degree)

    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

%     figure; 
%     plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); 
    errList = [errList; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
end

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(degreeList, errList, '*-');
xlabel('degree value'), ylabel('number of misclass');