function Coeff = solve_tl( X , lambda , display  )

% This routine solves the following trace lasso optimization problem for
% each data point y=x_i (i=1,...,n) based on
% X=[x_1,...,x_{i-1},x_{i+1},...x_n].

% min 0.5 * ||y-Xw||_2^2 + lambda * ||X*Diag(w)||_*

% if nargin<2
%     norm_x = norm(X,2) ;
%     lambda = 1/(sqrt(n)*norm_x) ;
% end

if nargin<3
    display = false ;
end
if nargin < 2
    lambda = 0.0001 ;
end
[dim,num] = size(X) ;
% for i = 1 : num
%    X(:,i) = X(:,i) / norm(X(:,i)) ; 
% end

Coeff = zeros( num , num ) ;
tol = 1e-5;
maxIter = 200 ; %1e6 500
tol2 = 1e-4 ;
rho0 = 1.1 ;
max_mu = 1e10 ;
for i = 1 : num
    y = X(:,i) ;
    if i == 1
        ind = 2:num ;
    elseif i == num
        ind = 1:num-1 ;
    else
        ind = [1:i-1 i+1:num] ;
    end
    riX = X(:,ind) ; 
    XtX = riX'*riX ;
    diagXtX = diag(diag(XtX)) ;
    Xty = riX'*y ;
    %% Initializing optimization variables
    mu = 0.1 ;%2.5/min(num,dim) ;
    w = zeros(num-1,1) ;
%     w = inv(riX'*riX+lambda*eye(num-1,num-1)) * riX' * y ;
    Z = zeros(dim,num-1) ;
    Y = zeros(dim,num-1) ;
    iter = 0 ;    
    while iter<maxIter
        iter = iter + 1; 
        w_old = w ;
        Z_old = Z ;       
        
        %update Z
        temp = riX*diag(w) - Y/mu ;
        [U,sigma,V] = svd(temp,'econ');
        sigma = diag(sigma);
        svp = length( find( sigma>lambda/mu ) ) ;
        if svp>=1
            sigma = sigma(1:svp)-lambda/mu ;
        else
            svp = 1 ;
            sigma = 0 ;
        end
        Z = U(:,1:svp)*diag(sigma)*V(:,1:svp)' ;

        %udpate w        
        A = XtX + mu*diagXtX ;
        b = Xty + diag((Y+mu*Z)'*riX) ;
        w = A\b ;        
        
        ymxw = y - riX*w ;
        leq = Z - riX*diag(w) ;        
        stopC = max(max(abs(leq))) ;        
        if display && (iter==1 || mod(iter,1)==0 || stopC<tol)
            err = norm(ymxw) ;
            reg = nuclearnorm(riX*diag(w)) ;
            obj(iter) = 0.5*err^2 + lambda*reg + trace(Y'*leq)+mu*norm(leq,'fro')/2 ;
%             obj(iter) = 0.5*err^2 + lambda*reg ;
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
                ',rank=' num2str(rank(riX*diag(w))) ',stopALM=' num2str(stopC,'%2.3e') ...
                ',err=' num2str(err) ',norm=' num2str(reg) ',obj=' num2str(obj(iter)) ]);
        end
        
        if stopC<tol
            break;
        else
            Y = Y + mu*leq;
            if max( max(abs(w-w_old)) , max(max(abs(Z-Z_old))) ) > tol2
                rho = rho0 ;
            else
                rho = 1 ;
            end
            mu = min(max_mu,mu*rho);
%             mu = min(max_mu,mu*rho(iter));
        end 
    end    
    Coeff(ind,i) = w ;
end


