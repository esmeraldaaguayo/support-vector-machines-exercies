load iris

%
% use RBF kernel
%

% tune the sig2 while fix gam
%
disp('RBF kernel')
type = 'c';
gamlist = [0.01,0.1,1,10,100,1000,10000]; 
% sig2list=[0.01, 0.1, 1, 5, 10, 25];
% sig2list=[0.001, 0.01, 0.1, 2, 11, 15, 20];
sig2 = 2;

errlist=[];

for gam=gamlist
     disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
     [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
     plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); 
    errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)        
end


%
% make a plot of the misclassification rate wrt. sig2
%
% figure;
% plot(log(sig2list), errlist, '*-'), 
% xlabel('log(sig2)'), ylabel('number of misclass'),

figure;
plot(log(gamlist), errlist,'*-');
xlabel('log(gam)'); 
ylabel('number of misclass');
