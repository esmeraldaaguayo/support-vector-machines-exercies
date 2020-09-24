load iris;

% gam= 1.466;
% sig2=1.995;

% gam= 0.090;
% sig2=0.528;

gam= 5.909 ;
sig2=0.308 ;
bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');