% approximation error vs actual error
clc, clear
n = 200;
A = genDecay(n);
p = 0;
q = 0;
nSamples = 150;
errs = zeros(1, nSamples);
mins = zeros(1, nSamples);
for l = 1:nSamples
    [U, S, Vt, Q] = rsvd(A, l, p, q);
    err = norm((eye(n)-Q*Q')*A);
    errs(l) = log10(err);
    
    s = svd(A);
    min = s(l+1);
    mins(l) = log10(min);
end

hold on

plot(1:nSamples, mins);
plot(1:nSamples, errs);

legend({'$\log_{10}(\sigma_{l+1})$', '$\log_{10}(e_l)$'},'Interpreter', 'latex')
ylabel('对数误差的量级')
xlabel({'$l$'},'Interpreter','latex');
title('真实误差和理论下界对比')
set(gcf,'unit','inch','position',[10 5 9 4]);
saveas(gcf, './figures/decay', 'jpg')


