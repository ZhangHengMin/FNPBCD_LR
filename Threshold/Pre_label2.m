function Y = Pre_label2(num_c,num_train, r,c)
% gnd:
% num_l: Y_label: 
Y=zeros(r,c);     
r_start = (num_c-1)*num_train + 1;
r_end = num_c*num_train;
Y(r_start:r_end,:) = 1;


