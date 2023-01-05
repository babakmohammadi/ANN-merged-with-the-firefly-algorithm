function Cost = Cost_ANN_FA(X,Xtr,Ytr,Network)
%% cost function
    Network = ConsNet_Fcn(Network,X);
    YtrNet = sim(Network,Xtr);
    Cost = mse(YtrNet - Ytr);
end
