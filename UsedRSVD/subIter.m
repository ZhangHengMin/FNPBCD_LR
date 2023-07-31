function Q = subIter(A, Y0, nIter)
Q = orthoBasis(Y0);
for i = 1:nIter
    Z = orthoBasis(A'*Q);
    Q = orthoBasis(A*Z);
end
end