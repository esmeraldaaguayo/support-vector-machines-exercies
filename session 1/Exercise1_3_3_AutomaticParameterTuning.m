load iris



RecordSimplex =[];
RecordGrid =[];

for i=1:1
%     tic
%     [gam1 ,sig21 , cost1 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
%     RecordSimplex =[RecordSimplex; [gam1, sig21, cost1]];
%     timeElapsed = toc
%     print(timeElapsed)
     tic
     [gam2 ,sig22 , cost2 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
     RecordGrid =[RecordGrid; [gam2, sig22, cost2]];
     timeElapsed1 = toc
     print(timeElapsed1)
end

