function [Network2] = TrainUsing_FA_Fcn(Network,Xtr,Ytr)

%% Problem Definition
IW = Network.IW{1,1}; IW_Num = numel(IW);
LW = Network.LW{2,1}; LW_Num = numel(LW);
b1 = Network.b{1,1}; b1_Num = numel(b1);
b2 = Network.b{2,1}; b2_Num = numel(b2);

TotalNum = IW_Num + LW_Num + b1_Num + b2_Num;

nVar = TotalNum;



VarSize=[1 nVar];

VarMin=-1;
VarMax= 1;
CostFunction=@(x) Cost_ANN_FA(x,Xtr,Ytr,Network);

%% Firefly Algorithm Parameters

MaxIt=100;         % Maximum Number of Iterations

nPop=30;            % Number of Fireflies (Swarm Size)

gamma=1;            % Light Absorption Coefficient

beta0=2;            % Attraction Coefficient Base Value

alpha=0.2;          % Mutation Coefficient

alpha_damp=.98;    % Mutation Coefficient Damping Ratio

delta=0.05*(VarMax-VarMin);     % Uniform Mutation Range

m=2;

if isscalar(VarMin) && isscalar(VarMax)
    dmax = (VarMax-VarMin)*sqrt(nVar);
else
    dmax = norm(VarMax-VarMin);
end

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Cost=[];

% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Fireflies
for i=1:nPop
   pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
   pop(i).Cost=CostFunction(pop(i).Position);
   
   if pop(i).Cost<=BestSol.Cost
       BestSol=pop(i);
   end
end


BestCost=zeros(MaxIt,1);

%% Firefly Algorithm Main Loop

for it=1:MaxIt
    
    newpop=repmat(firefly,nPop,1);
    for i=1:nPop
        newpop(i).Cost = inf;
        for j=1:nPop
            if pop(j).Cost < pop(i).Cost
                rij=norm(pop(i).Position-pop(j).Position)/dmax;
                beta=beta0*exp(-gamma*rij^m);
                e=delta*unifrnd(-1,+1,VarSize);
                                
                newsol.Position = pop(i).Position ...
                                + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
                                + alpha*e;
                
                newsol.Position=max(newsol.Position,VarMin);
                newsol.Position=min(newsol.Position,VarMax);
                
                newsol.Cost=CostFunction(newsol.Position);
                
                if newsol.Cost <= newpop(i).Cost
                    newpop(i) = newsol;
                    if newpop(i).Cost<=BestSol.Cost
                        BestSol=newpop(i);
                    end
                end
                
            end
        end
    end
    
    
    pop=[pop
         newpop];  
    
    
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    
    pop=pop(1:nPop);
    
    
    BestCost(it)=BestSol.Cost;
    
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    
    alpha = alpha*alpha_damp;
    
end

%% Results
Network2 = ConsNet_Fcn(Network,BestSol.Position);
figure;
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
