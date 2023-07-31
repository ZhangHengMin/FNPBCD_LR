function [L,err,iter,Time] = ReRMCrFig_FNPBCD(Y, A, lamda, kk, qq, para)
%%
%   minimize lamda*||L||_Sq1^Sq1 + ||S||_q2^q2 +  mu/2*||Y-L-S||_F^2
%
% Inputs
%	1=>q1,q2=>0
%	Ltrue, Strue: for debug, for calculation of errors
%   L0,S0: intialization
% Outputs
%	L,S: the recovery
%	out.el, out.es: the error with respect to the true

[m, n] = size(Y);
%Initialize
if(nargin==6 & isfield(para, 'MAX_ITER'))
    MAX_ITER = para.MAX_ITER;
else
    MAX_ITER = 2e3;
end

if(nargin==6 & isfield(para, 'TOL'))
    ABSTOL = para.TOL;
else
    ABSTOL = 1e-5;
end

if(nargin==6 & isfield(para, 'L0'))
    L = para.L0;
else
    L = zeros(m,n);
end

if(nargin==6 & isfield(para, 'S0'))
    S = para.S0;
else
    S = zeros(m,n);
end

idx = find(A==1);
idx1 = find(A~=1);
%
mu  = 300/norm(Y); %
phi = kk;
%
r = para.MAX_RANK;
q = 1;   % Power iterations
p = 5;   % Oversampling parameter

totalTime = 0;
for iter = 1:MAX_ITER
    timeFlag = tic;
    Lm1 = L;
    %Sm1 = S;

    % for acceleration of the algorithm
    mu = mu * 1.1; 

    %L-update
    X = (mu*(Y-S)+phi*L) / (mu+phi);
    [U,V,D] = rsvd(X,r,q,p); % Randomized SVD
    s = diag(V);
    v = shrinkage_Lq(s, qq, lamda, mu+phi);
    indx = find(v);
    L = U(:,indx)*diag(v(indx))*D(:,indx)';


    %% S-update
    T = (mu*(Y-L)+phi*S)/(mu+phi);
    S = reshape(shrinkage_Lq(T(:), qq, 1, mu+phi), m, n);
    S(idx1) = T(idx1);


    totalTime = totalTime + toc(timeFlag);
    Time(iter) = totalTime;
    % Check for convergence
    err(iter) = norm(L-Lm1,'fro')/norm(L,'fro');
    if err(iter) < ABSTOL
        break;
    end

end


