function [X,E,out] = mLRR(fun,A,D,lambda,paras)
%   minimize \gamma/2*||A*Z +E - D ||_2^2
%      + \lambda*|| X ||_nonconvex + || Y ||_nonconvex s.t.,  Z = X,  E = Y.
%    A=D,E,D\in R^{m x n} Z,X\in R^{n x n}
% Inputs
%	A,D,lambda,non_f,paras
% Outputs
%   X,E

% Convergence setup
MAX_ITER = 500;
ABSTOL = 1e-3;
%
[m, n] = size(D);
%
rho  = 1e6;
beta = 1e6;

Z = zeros(n,n);
%Y = zeros(m,n);
%
 
X = eye(n,n);
E = zeros(m,n);
La1 = ones(n,n);
La2 = zeros(m,n);
%
% gamma = 0.15*norm(D);        % changable very important                     % 0.02*norm(y);
% mu = 0.25;                  % changable very important
gamma = 0.25*norm(D);                  % changable very important                     % 0.02*norm(y);
mu = 2.0;%
%
p = paras.p;
% out.et = [];out.e = [];
An = A./norm(A,2);
[v,d] = eigs(An'*An);
display = 1;
hfun = str2func( [ fun ''] ) ;
tic;
%
for iter = 1:MAX_ITER
    %%
    if gamma < beta                                             % for acceleration of the algorithm
        gamma = gamma * 1.1;
    end
    
    if mu < rho                                               % for acceleration of the algorithm
        mu = mu * paras.rho;
    end
     
    %%
    E_pre = E;
    % update X, Y
    X_temp = Z + La1/mu;
    Y_temp = E + La2/mu;
    %
    [U, S, V] = svd(X_temp,0);
    s = diag(S);
    xpara = lambda/mu;
    for i = 1:length(s)
        s1(i) = Sp_rank(fun,s(i), xpara, p);
    end;
    X = U*diag(s1)*V';
    %
    ypara = 1/mu;
    for i = 1:n
        yy = Y_temp(:,i);
        s2(:,i) = Sp_rank(fun, norm(yy), ypara, p).*yy;
        columns(i) = norm(s2(:,i));
    end;
    Y = s2;
    
    % update Z, E
    Z_temp = mu*(X-1/mu*La1)+gamma*A'*(D-E);
    Z = inv(gamma*A'*A+mu*eye(n)) * Z_temp;
    %
    E_temp = mu*(Y-1/mu*La2)+gamma*(D-A*Z);
    E = 1/(gamma+mu) * E_temp;
    
    % update dual variables
    La1 = La1 + mu*(Z - X);
    La2 = La2 + mu*(E - Y);
    
        % some property
         if display
           mu_k1 = min(1.1*mu, rho); c_2 = 2*gamma^2*max(d(:));
           Error_value = (mu_k1+mu)/(2*mu^2)*c_2*norm(E-E_pre, 'fro')^2;
           obj_value1 = sum(hfun(s1,p,lambda)) + sum(hfun(columns,p,1)) + gamma/2*norm(A*Z+E-D, 'fro')^2;
           obj_value2 = -1/(2*mu)*(norm(La1, 'fro')^2 + norm(La2, 'fro')^2) + mu/2*(norm(Z-X+La1/mu,'fro')^2 + norm(E-Y+La2/mu, 'fro')^2);
           obj_original(iter) = obj_value1 + obj_value2;
           obj_potential(iter) = obj_original(iter) + Error_value;
         end
    
    
     err1(iter) = norm(Z-X)/sqrt(n);
     err2(iter) = norm(E-Y)/sqrt(m);
    
    % Check for convergencel
    if (norm(Z-X) < sqrt(n)*ABSTOL) && (norm(E-Y) < sqrt(m)*ABSTOL)
        break;
    end
    
end
out.obj = obj_potential;
out.err1 = err1;
out.err2 = err2;
end

