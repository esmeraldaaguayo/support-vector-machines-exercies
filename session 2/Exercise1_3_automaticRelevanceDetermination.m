X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;

figure;
plot(Y);

gam = 0.2632;
sig2 = 0.2218;

[ selected , ranking, costs] = bay_lssvmARD ({X, Y, 'f', gam , sig2 });

OptX1 = X(:,1);

OptX2 = [X(:,1),X(:,2)];

OptX3 = [X(:,1),X(:,2), X(:,3)];

% [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
% plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

position = [1,2,3];

plotData = [position;costs];
figure;
bar(costs);
hold on;
% plot(position);
