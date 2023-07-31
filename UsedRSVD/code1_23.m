

%%% reconstruc image by classical SVD, adn Random SVD
clear all, close all, clc
A=imread('dog.jpg');
X=double(rgb2gray(A));
subplot(141); 
imagesc(X); axis off; colormap gray;
tic;
[U,S,V] = svd(X,'econ');    % Deterministic SVD
toc
r = 400; % Target rank
q = 1;   % Power iterations
p = 5;   % Oversampling parameter
tic
[rU,rS,rV] = rsvd(X,r,q,p); % Randomized SVD
toc
r = 400; % Target rank
q = 1;   % Power iterations
p = 5;   % Oversampling parameter
tic
[rU1,rS1,rV1,W1] = rsvd1(X,r,q,p); % Randomized SVD
toc
%% Reconstruction
XSVD = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';     % SVD approx.
errSVD = norm(X-XSVD,2)/norm(X,2)

XrSVD = rU(:,1:r)*rS(1:r,1:r)*rV(:,1:r)'; % rSVD approx.
errrSVD = norm(X-XrSVD,2)/norm(X,2)

XrSVD1 = rU1*rS1*rV1; % rSVD approx.
errrSVD1 = norm(X-XrSVD1,2)/norm(X,2)


subplot(142); 
imagesc(XSVD); axis off; colormap gray; 
subplot(143);
imagesc((XrSVD));  colormap gray;
subplot(144);
imagesc((XrSVD1)); colormap gray;
axis off; 