function A = genDecay(n)
U = eye(n);
S = -0.75.^(1:n);
Vt = eye(n);
A = U*diag(S)*Vt;
end