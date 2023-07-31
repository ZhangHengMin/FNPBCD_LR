function nnorm = nuclearnorm( X )

% ¼ÆËã¾ØÕóXµÄnuclear norm, |X|_*

s = svd(X) ;
nnorm = sum( s ) ;

% index = 1 : 7 ;%length(s) ;
% nnorm = sum( s(index) ) ;