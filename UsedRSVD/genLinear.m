function A = genLinear(n, k)
U = eye(n);
S = [(1:k),zeros(1,n-k)];
Vt = eye(n);
A = U * diag(S) * Vt;
end