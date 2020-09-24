
load iris

% 1.3.2 Tuning parameters using validation
gamList=[0.001,0.01,0.1,1,10,100,1000];
sig2List=[0.001,0.01,0.1,1,10,100,1000];
% gamList=[0.001,1,10,1000];
% sig2List=[0.001,1,10,1000];
% gamList=[1,10];
% sig2List=[1,10];

error = 1;
perfVector =[];
allVector=[];
minGam = 1000;
minSig2 =1000;
zeroVector=[];

for gam=gamList
    for sig2=sig2List
%          perf = rsplitvalidate ({ Xtrain , Ytrain ,'c', gam, sig2,'RBF_kernel'}, 0.80 , 'misclass');
%          perf = crossvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, 10, 'misclass');
         perf = leaveoneout ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, 'misclass');
        
        allVector = [allVector; [gam,sig2,perf]];
        perfVector = [perfVector, perf];
        if perf<error
            error = perf;
            minGam = gam;
            minSig2 = sig2;
        end
        if perf == 0
            zeroVector = [zeroVector; [gam, sig2, perf]]
        end
    end
end
z = perfVector;

% figure;
% X2 = [x,y];
% Z= meshgrid(z);
% mesh(Z);

figure;
plot(z);
xlabel('Function Evaluations');
ylabel('Misclassification Error');

% figure;
% [X1,Y1] = meshgrid(x,y);
% [Z1] =meshgrid(z);
% surf(X1,Y1,Z1);

% figure;
% scatter3(x,y,z);
% xlabel('gam');
% ylabel('sig2');
% zlabel('misclass error');





