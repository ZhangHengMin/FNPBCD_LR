function [D, class_test] = spnf_classify(Tr, Te, alpha, opts, fun_num)
%   Summary of this function goes here
%   Detailed explanation goes here

nnClass = opts.nnClass;
num_train = opts.sele_num;
row = opts.row;
col = opts.col;
D = 0;
for i = 1 : nnClass
    MSK = Pre_label2(i,num_train, size(alpha,1), size(alpha,2));
    alpha_si = alpha.*MSK;
    Si = Tr*alpha_si;
    %D(i,:) = sqrt(sum((Te - Si).^2));
    alpha_norm2 = sqrt(sum(alpha.^2));
    %D(i,:) = sqrt(sum((Tr*alpha - Si).^2));
    Di = Te - Si;
    for j = 1 : size(Te,2)
        Dim = reshape(Di(:,j), [row, col]);
        ss = svd(Dim);
        switch(fun_num)
            case 1
                D(i,j) = sum(ss)/alpha_norm2(j);
            case 2
                D(i,j) = sum(ss.^(2/3))/alpha_norm2(j);
            case 3
                D(i,j) = sum(ss.^(0.5))/alpha_norm2(j);
        end
    end
end
%
[~,class_test] = min(D);
%
end

