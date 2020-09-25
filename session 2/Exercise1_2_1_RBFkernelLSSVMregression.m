% Generate data
X = ( -3:0.01:3)';
Y = sinc (X) + 0.1.* randn ( length (X), 1);

% Separate into datasets
Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);

errVector= [];
gamList = [10, 1000, 1000000];
sig2List = [0.01, 1, 100];
type = 'function estimation';
for gam=gamList
    for sig2=sig2List
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
        Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
        err = immse(Ytest,Yt);
        errVector = [errVector; err];
        figure;
        plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
        hold on;
        
    end
end

simVector = [];
for i=1:10
    tic;
    type = 'function estimation';
    [simGam, simSig2, cost1]=tunelssvm ({ Xtrain , Ytrain ,type, [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
    time1 = toc;
    simVector = [simVector; [simGam, simSig2, cost1, time1]];
end

gridVector = [];
for i=1:10
    tic;
    type = 'function estimation';
    [gridGam, gridSig2,cost2]=tunelssvm ({ Xtrain , Ytrain ,type, [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'mse'});
    time2 = toc;
    gridVector = [gridVector; [gridGam, gridSig2,cost2, time2]];
end

