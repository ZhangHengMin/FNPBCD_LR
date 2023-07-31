function [U, S, Vt, Q] = rsvd1(A, k, p, q)
l = k + p;
Q = findRange(A, l, q);
B = Q'*A;
[UB, S, V] = svd(B);
Vt = V';
U = Q*UB;
U = U(:,1:k);
S = S(1:k, 1:k);
Vt = Vt(1:k, :);     
end