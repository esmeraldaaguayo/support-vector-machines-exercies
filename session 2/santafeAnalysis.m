load ('santafe.mat');

% figure;
% plot(Z);
% xlabel 'time';
% ylabel 'laser signal';
% 
% figure;
% join = [Z;Ztest];
% plot(join);
% xlabel 'time';
% ylabel 'laser signal';
% 
% figure;
% plot(Ztest);
% xlabel 'time';
% ylabel 'laser signal';

% orderList = [5,7,10,13,15];
% orderList = [10,15,20,30,40,50,60];
orderList = [50];

ZsubTrain = [Z(1:550,1);Z(651:end,1)];
ZsubValidate = Z(551:650, 1);
errVector=[];
modelError = Inf;
optOrder = 0;
optGam = 0;
optSig2 =0;

for order=orderList
    X = windowize (ZsubTrain, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
    
    model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
    costFun = 'crossvalidatelssvm';
    [gam, sig2] = tunelssvm (model , 'simplex', costFun , {10 , 'mae';});
    
    [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
    
    Xs = ZsubTrain(end - order +1: end , 1);
    nb = 100;
    Yp = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
    err = immse(ZsubValidate,Yp);
    errVector = [errVector; err];
    
    %select best model
    if err< modelError
        modelError = err;
        optOrder = order;
        optGam = gam;
        optSig2 = sig2;
    end
    
end


X = windowize (ZsubTrain, 1:( optOrder + 1));
Y = X(:, end);
X = X(:, 1: optOrder );

[alpha , b] = trainlssvm ({X, Y, 'f', optGam , optSig2 });
 
Xs = Z(end - optOrder +1: end , 1);
nb = 200;
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
 
figure ;
hold on;
plot (Ztest , 'k');
plot ( prediction , 'r');
xlabel 'time';
ylabel 'a.u.';
hold off;


