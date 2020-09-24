X1 = randn (50 ,2) + 1;
X2 = randn (51 ,2) - 1;
Y1 = ones (50 ,1);
Y2 = -ones (51 ,1);
input = [X1; X2];
target = [Y1; Y2];
% disp (target);
figure ;
hold on;
plot (X1 (: ,1) , X1 (: ,2) , 'ro');
plot (X2 (: ,1) , X2 (: ,2) , 'bo');
hold off;

net = perceptron;
net = configure(net,input',target');
 
net = adapt(net,input',target');
figure ;
hold on;
plot (X1 (: ,1) , X1 (: ,2) , 'ro');
plot (X2 (: ,1) , X2 (: ,2) , 'bo');
plotpc(net.IW{1},net.b{1});
hold off;



