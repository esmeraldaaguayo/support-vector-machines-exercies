load logmap.mat
% 
% figure;
% plot(Z);
% xlabel 'time';
% ylabel 'a.u.';
% 
% figure;
% join = [Z;Ztest];
% plot(join);
% xlabel 'time';
% ylabel 'a.u.';
% 
% figure;
% join2 = [Z;Ztest_withoutnoise];
% plot(join2);
% xlabel 'time';
% ylabel 'a.u.';
% 
% figure;
% plot(Ztest);
% xlabel 'time';
% ylabel 'a.u.';

% order =10;
% orderList = [5,6,7,8,9,10];
% orderList = [5,7,10,13,15];
orderList = [10,13,15,20,30,40,50];
ZsubTrain = Z(1:100,1);
ZsubValidate = Z(101:end, 1);
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
    costFun = 'rcrossvalidatelssvm';
    wFun = 'whuber';
    [gam, sig2] = tunelssvm (model , 'simplex', costFun , {10 , 'mae';}, wFun );
    
    [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
    
    Xs = ZsubTrain(end - order +1: end , 1);
    nb = 50;
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
nb = 50;
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
 
figure ;
hold on;
plot (Ztest , 'k');
plot ( prediction , 'r');
xlabel 'time';
ylabel 'a.u.';
hold off;


