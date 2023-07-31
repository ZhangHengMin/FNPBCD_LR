clear all;
close all;  
addpath('DATA');
addpath('UsedRSVD');
addpath('Threshold');
addpath('CoEV');

dataset='COIL20_32x32'; %2580x700
load([dataset,'.mat']);
fea = double(fea); nnClass =  length(unique(gnd));
num_Class = []; sele_num = 30;
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))];
end
Sefea = [];
Segnd = [];
for j = 1:nnClass
    idx = find(gnd==j);
    Sefea = [Sefea;fea(idx((1:sele_num)),:)];
    Segnd= [Segnd;gnd(idx((1:sele_num)))];
end
Sefea = Sefea'/256;
Segnd = Segnd';
%%
X = double(Sefea);
X = X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);%m*N
gnd = Segnd';
gnd = double(gnd);
K = max(gnd);

param.nnClass = nnClass;
param.sele_num = sele_num;
param.TOL =1e-3;
param.MAX_ITER = 1e3;
param.MAX_RANK = nnClass;

Promethods = {'ReLRR1_FNPBCD12','ReLRR1_FNPBCD23'};
for fun_num = 1 : length(Promethods)
    mymethod = Promethods{fun_num};
    disp([' choosemethod = '  mymethod]);

    switch (mymethod)

        case 'ReLRR1_FNPBCD12'
            lamParas =  [0.01 0.1 1.0 3.0 5.0 8 10];
            for iparas = 1  :  length(lamParas)
                lamdas = lamParas(iparas);
                disp([' lambda = ' num2str(lamdas)]);
                kkParas =  [10 20 30 40 50 60 70];
                for kiparas = 1  :  length(kkParas)
                    param.kk = kkParas(kiparas);

                    for i = 1 : size(X,2)
                        X(:,i) = X(:,i) /norm(X(:,i)) ;
                    end

                    tic;
                    [Z, err, E,iter, RunTime] = ReLRR1_ErrorFaPBCD(X, X, lamdas, 0.5, 0.5, param);
                    time_cost = toc;

                    W = Z;
                    for ic = 1 : size(Z,2)
                        W(:,ic) = Z(:,ic)/(max(abs(Z(:,ic)))) ;
                    end 
                    L = Selection(W, K) ;
                    W2 = ( abs(L) + abs(L') ) / 2 ;

                    idx = clu_ncut(W2,K);
                    acc = compacc(idx',gnd);
                    score = MutualInfo(idx', gnd);
                    disp([' mu = ' num2str(param.kk), ' seg acc= ' num2str(acc*100), ' seg nmi= ' num2str(score*100), ' cost time= ' num2str(time_cost), ' num iter= ' num2str(iter)]);
 

                end 

            end 



        case 'ReLRR1_FNPBCD23'
            lamParas =  [0.01 0.1 1.0 3.0 5.0 8 10];
            for iparas = 1  :  length(lamParas)
                lamdas = lamParas(iparas);
                disp([' lambda = ' num2str(lamdas)]);
                kkParas =  [10 20 30 40 50 60 70];
                for kiparas = 1 :  length(kkParas)
                    param.kk = kkParas(kiparas);

                    for i = 1 : size(X,2)
                        X(:,i) = X(:,i) /norm(X(:,i)) ;
                    end

                    tic;
                    [Z, err, E,iter, RunTime] = ReLRR1_ErrorFaPBCD(X, X, lamdas, 2/3, 2/3, param);
                    time_cost = toc;

                    W = Z;
                    for ic = 1 : size(Z,2)
                        W(:,ic) = Z(:,ic)/(max(abs(Z(:,ic)))) ;
                    end
                    L = Selection(W, K) ;
                    W2 = ( abs(L) + abs(L') ) / 2 ;

                    idx = clu_ncut(W2,K); 
                    acc = compacc(idx',gnd);
                    score = MutualInfo(idx', gnd);


                    disp([' mu = ' num2str(param.kk), ' seg acc= ' num2str(acc*100), ' seg nmi= ' num2str(score*100), ' cost time= ' num2str(time_cost), ' num iter= ' num2str(iter)]);
 

                end 

            end 

    end

end