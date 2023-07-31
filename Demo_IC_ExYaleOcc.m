clear all;
close all;
 
addpath('DATA');
addpath('UsedRSVD');
addpath('Threshold');
%
%
sele_num = 7;
nnClass        =   38;
load('YaleBIllu5_96x84.mat')
Tr_DAT55 = Tr_DAT;
Tt_DAT55 = Tt_DAT;
trls55 = trls;
clear Tr_DAT; clear Tt_DAT;
load('YaleBFeifei_96x84.mat')
clear Tr_DAT; clear trls;
[m, k1, l1] = size(Tr_DAT55);
[m, k2l2] = size(Tt_DAT);
Tr_DAT = reshape(Tr_DAT55, [m, k1*l1]);
% Tr_DAT   =   Tr_DAT/256;
% Tt_DAT   =   Tt_DAT/256;
trls = trls55;
Image_row_NUM = 96; Image_column_NUM = 84;

param.row = Image_row_NUM;
param.col = Image_column_NUM;
tr_dat  =  Tr_DAT;
tt_dat  =  Tt_DAT;
% normalization
tr_descr  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat,1),1]) );
tt_descr  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tr_dat,1),1]) );
Train_Ma = tr_descr; Train_Lab = trls';
Test_Ma  = tt_descr; Test_Lab = ttls';

param.nnClass = nnClass;
param.sele_num = sele_num;
param.TOL =1e-5;
param.MAX_ITER = 1e3;
param.MAX_RANK = nnClass;

Promethods = {'ReRMR1_FNPBCD12', 'ReRMR1_FNPBCD23'};
for fun_num = 1 : length(Promethods)
    mymethod = Promethods{fun_num};
    disp([' choosemethod = '  mymethod]);

    switch (mymethod)

         
        case 'ReRMR1_FNPBCD12'
            lam_num =  [0.001 0.01 0.1 1.0 3.0 5 10  20 30 40 50];
            %
            for kk = 1 : length(lam_num)
                lambda = lam_num(kk);
  
                tic;
                [L,err,iter, RunTime] = ReRMR1_TimeFNPBCD(Test_Ma, Train_Ma, lambda, 0.5, 0.5, param);
                time_cost = toc; 

                [D, class_test] = spnf_classify(Train_Ma, Test_Ma, L, param, 3);
                acc_test = sum(Test_Lab == class_test')/length(Test_Lab)*100;
                disp([ ' lambda = ' num2str(lambda), ' acc_result= ' num2str(acc_test), ' time_cost = ' num2str(time_cost), ' ITERNum= ' num2str(iter)]);
            end

        

        case 'ReRMR1_FNPBCD23'


            lam_num =  [0.001 0.01 0.1 1.0 3.0 5 10  20 30 40 50];
            %
            for kk = 1 : length(lam_num)
                lambda = lam_num(kk);
 
                tic;
                [L,err,iter, RunTime] = ReRMR1_TimeFNPBCD(Test_Ma, Train_Ma, lambda, 2/3, 2/3, param);
                time_cost = toc; 

                [D, class_test] = spnf_classify(Train_Ma, Test_Ma, L, param, 2);
                acc_test = sum(Test_Lab == class_test')/length(Test_Lab)*100;
                disp([ ' lambda = ' num2str(lambda), ' acc_result= ' num2str(acc_test), ' time_cost = ' num2str(time_cost), ' ITERNum= ' num2str(iter)]);
            end


    end

end

 