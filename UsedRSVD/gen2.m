% usage of subspace iter 
clc, clear
hold on
k = 29;
n = 50;
nIter = 3;
h = {};
A = genLinear(n, k);

[~, STrue, ~] = svd(A);
STrue = diag(STrue);
STrue = STrue./max(STrue);
x = 1:length(STrue);
h{1} = scatter(x, STrue, 16, 'x');
plot(x, STrue);

for q = 1:nIter
    ANew = (A*A')^(q)*A;
    [~, SNew, ~] = svd(ANew);
    SNew = diag(SNew);
    SNew = SNew./max(SNew);
    h{q+1} = scatter(x, SNew, 16, '*');
%     legend(h, sprintf('q=%d',q))
    plot(x, SNew);
end

legend([h{1}, h{2}, h{3}, h{4}], '真实的奇异值', 'q=1', 'q=2', 'q=3')


ylabel('归一化后的奇异值')
xlabel({'$\sigma_i$'},'Interpreter','latex');
title('q次迭代后归一化的奇异值')
set(gcf,'unit','inch','position',[10 5 9 3.5]);
saveas(gcf, './figures/subspace', 'jpg')
