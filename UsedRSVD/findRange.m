function Q = findRange(A, nSamples, nIter)
[~, n] = size(A);
O = randn(n, nSamples);
Y = A*O;
if nIter
    Q = subIter(A, Y, nIter);
else
    Q = orthoBasis(Y);
end