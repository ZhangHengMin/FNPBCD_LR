%===================================================
% Fast SVD 
% function [U,S, V]=rsvd (X, r,q,p)
%
% usage : 
%
%  input:
%  * X : matrix whose SVD we want
%  * r :  trunction (rank ) for example 400
%  * q :  power of iteration for eample 1
%  * p  : oversmpling  parameter for example 5
%  number of components to keep
%
%  output:
%  * U,S,V :  right eigenvector, singular value , left eigenvector


function [U,S, V]=rsvd (X, r, q, p)
 
ny=size(X,2);
P=randn(ny,r+p); 
Z=X*P;
for k=1:q
   Z=X*(X'*Z) ;
    
end
[Q E]=qr(Z,0);

Y=Q'*X; 
[UY, S ,V]=svd(Y, 'econ');
U=Q*UY;
end