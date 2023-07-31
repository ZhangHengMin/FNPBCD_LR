% demo in image decompress
clc, clear;
hold on
img = imread('peppers.png');                              
A = rgb2gray(img);
A = double(A);
[m, ~] = size(A);
k = 40;
q = 0;

[U, S, V] = svd(A);
U = U(:,1:k);
S = S(1:k, 1:k);
Vt = V';
Vt = Vt(1:k, :);
MPepper = U*S*Vt;
I = uint8(MPepper);
subplot(2, 2, 1)
imshow(I)
title(sprintf('k=%d的标准奇异值分解', k))

ls = [1 00 100];
i = 2;
for l = ls
    [U, S, Vt, Q] = rsvd(A, k, l, q);
    MPepper = U*S*Vt;
    I = uint8(MPepper);
    subplot(2, 2, i);
    i = i+1;
    imshow(I)
    title(sprintf('k=%d, l=%d的随机奇异值分解', k, l))
end

saveas(gcf, 'figures/peppers', 'jpg')

