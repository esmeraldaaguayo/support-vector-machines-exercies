% ROC curve
load iris;
 
gam= 1.466;
sig2=1.995;

% Train the classification model.
[alpha , b] = trainlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'});

% Classification of the test data.
[Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, {alpha , b}, Xtest );

% Generating the ROC curve.
[tpr,fpr,thresholds] = roc( Yest,Ylatent );
% plotroc(Ytest,Ylatent);
[X1,Y1] = perfcurve(Ytest, Ylatent, 1);
plot (X1,Y1);


% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression')