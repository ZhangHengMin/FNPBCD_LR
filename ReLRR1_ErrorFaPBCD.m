function [L,stop, obj,iter,Time] = ReLRR1_ErrorFaPBCD(Y, A, lamda,q1,q2,para)
%%
%   minimize lamda*||L||_Sq1^Sq1 + ||S||_q2^q2 +  mu/2*||Y-A*L-S||_F^2
%
% Inputs
%	1=>q1,q2=>0
%	Ltrue, Strue: for debug, for calculation of errors
%   L0,S0: intialization
% Outputs
%	L,S: the recovery
%	out.el, out.es: the error with respect to the true

[m, n] = size(Y);
[m, n1] = size(A);
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
    L = zeros(n1,n);
end

if(nargin==6 & isfield(para, 'S0'))
    E = para.S0;
else
    E = zeros(m,n);
end

fun ='lp';
mu  = para.kk/norm(Y); %30
phi = 1e-5;
yita = 1.05*max(eig(A'*A));
r = para.MAX_RANK;
qq = 1;   % Power iterations
pp = 5;   % Oversampling parameter
totalTime = 0;
for iter = 1:MAX_ITER
    timeFlag = tic;

    Lm1 = L;
    %Sm1 = S;

    % for acceleration of the algorithm
    mu = mu * 1.1;

    %rand('state',0)
    % L-update
    
    temp_L = L-A'*(A*L+E-Y)/yita;
    MediaL = (mu*yita*temp_L+phi*L)/(mu*yita+phi);  
    [U,V,D] = rsvd(MediaL,r,qq,pp); % Randomized SVD 
    s = diag(V); 
    v = shrinkage_Lq(s, q1, lamda, mu*yita+phi);
    indx = find(v);
    L = U(:,indx)*diag(v(indx))*D(:,indx)';
 
    %% E-update
    EE5 = A*L-Y;
    MediaE = (mu*(Y-A*L)+phi*E)/(mu+phi);
    parfor i = 1:n
        yy = MediaE(:,i);
        EE(:,i) = Sp_rank(fun, norm(yy), 1/(mu+phi), q2).*yy; 
        eevalue(i) = norm(EE5(:,i));
    end
    E = EE;


    h1 = lamda*sum(v(indx).^q1);
    e1 =  sum(eevalue.^q2);
    %g1 = mu/2*norm(A*L+E-Y, 'fro')^2;
    obj(iter) = h1+e1;
    totalTime = totalTime + toc(timeFlag);
    Time(iter) = totalTime;
    % Check for convergence
    stop(iter) = norm(L-Lm1,'fro')/norm(L,'fro');
    if stop(iter) < ABSTOL
        break;
    end

end
