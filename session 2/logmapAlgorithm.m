load logmap.mat

orderList = [10,20,30,40,50];

for order=orderList
    X = windowize (Z, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
 
    model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
    costFun = 'rcrossvalidatelssvm';
    wFun = 'whuber';
    [gam, sig2] = tunelssvm (model , 'simplex', costFun , {10 , 'mae';}, wFun );
% 
%     gam =10;
%     sig2=10;
    [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
    Xs = Z(end - order +1: end , 1);
    nb = 50;
    prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
    figure ;
    hold on;
    plot (Ztest , 'k');
%     plot (Ztest_withoutnoise , 'k');
    plot ( prediction , 'r');
    hold off;
end

% order = 10;
