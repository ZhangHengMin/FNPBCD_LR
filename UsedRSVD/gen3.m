% error after subspace iteration
clc, clear;
hold on
img = imread('peppers.png');                              
A = double(rgb2gray(img)); 
[m, ~] = size(A);

l = 100;
q = 3; 
ls = 1:l;
qs = 0:q;
p = 0;

mins = zeros(1, l);
errs = zeros(q+1, l);

for l = ls
    [~, S, ~] = svd(A);
    min = S(l+1, l+1);
    mins(l) = min;
    
    for q = qs
        [~, ~, ~, Q] = rsvd(A, l, p, q);
        err = norm((eye(m)-Q*Q')*A);
        errs(q+1, l) = err;
    end
end
plot(ls(11:end), mins(11:end));
for q = qs
    plot(ls(11:end), errs(q+1,11:end));
end
legend('误差下界', 'q=0', 'q=1', 'q=2', 'q=3')
ylabel('误差量级')
xlabel({'$l$'},'Interpreter','latex');
title('引入迭代后的实际逼近误差与理论误差下界比较')
set(gcf,'unit','inch','position',[10 5 9 3.5]);
saveas(gcf, './figures/error_q', 'jpg')
