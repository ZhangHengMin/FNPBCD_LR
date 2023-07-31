function B = Selection( A , k )


% B = A ;
% fprintf('K=%d,----------------------------\n',k) ;
if k == 0
    B = A ;
else
    [a, ind] = sort(A,'descend') ;
    B = zeros( size(A) ) ;    
    if k < 1        
        % 此时的k表示保留最大的系数的比例
%         fprintf('K=%d,----------------------------\n',k) ;

        k = round( size(A,1) * k ) ;
    end
    for i = 1 : size(A,2)
        B( ind(1:k,i) , i ) = A( ind(1:k,i) , i ) ;
    end
end



% 大于0的全部设为1
% B(B>0) = 1 ;

% 下面的操作对YaleB来讲有提升效果，对Hopkins有下降效果

% for i = 1 : size(B,2)
%    B(:,i) = B(:,i) / max(B(:,i)) ;    
% end


