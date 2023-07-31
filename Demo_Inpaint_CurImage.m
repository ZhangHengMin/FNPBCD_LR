clear all;
close all;
addpath('DATA');
addpath('UsedRSVD');
addpath('Threshold');

%% curve mask
image = imread('ntu_longbao.jpg');

% Convert the image to double for adding noise
Xfull = im2double(image);
[m,n,c]=size(Xfull);
for ch = 1 :3
    [U1,S1,V1] = svd(Xfull(:,:,ch));
    s1 = diag(S1);
    s1(61:end) = 0;
    XTrsvs{ch} = s1;
    XTrfull(:,:,ch) = U1(:, 1:60)*diag(s1(1:60))*V1(:, 1:60)';
end
XTrfull = im2double(XTrfull);
ind = im2bw(imread('curvemask.jpg'));
mask(:,:,1)=ind;
mask(:,:,2)=ind;
mask(:,:,3)=ind;
Xmiss = XTrfull.*mask; 

% Define the noise density (percentage of pixels to be corrupted with noise)
noise_density = 0.1; % Set the desired noise density here (e.g., 0.1 for 10% noise)
 
% Generate random indices to add noise
num_pixels = numel(Xmiss);
num_noisy_pixels = round(noise_density * num_pixels);
noise_indices = randperm(num_pixels, num_noisy_pixels);

% Add sparse noise (replace the noisy pixels with random values between 0 and 1)
noisy_image = Xmiss;
noisy_image(noise_indices) = rand(size(noise_indices));
%%
param.TOL =1e-5;
param.MAX_ITER = 5e2;
param.MAX_RANK = round(0.10*min(m,n));

method = {'FNPBCDRMC12','FNPBCDRMC23'};
for numFun = 1  : 2
    disp([' ProposedMethod= ' num2str(method{numFun})]);

    switch(method{numFun})
        


        case 'FNPBCDRMC12'

            lamParas =  [10 20 30 40 50 80 100];
            for iparas = 1 :  length(lamParas)
                lamdas = lamParas(iparas);

                tic;
                for ch=1:3
                    [L, err, iter, Times] = ReRMCrFig_FNPBCD(noisy_image(:,:,ch), ind, lamdas, 1e-5, 1/2, param);
                    disp(['ch_num = ' num2str(ch), ' iter = ' num2str(iter)]);
                    Xhat(:,:,ch) = L; 
                end
                time_cost = toc;
                Xhat = max(Xhat,0);
                Xhat = min(Xhat,255);
                ourpsnr = PSNR(Xfull,Xhat,max(Xfull(:)));
                disp([' lambda = ' num2str(lamdas), ' ValuePSNR = ' num2str(ourpsnr), ' iter= ' num2str(iter), ' time= ' num2str(time_cost)]);
                 
            end

 


        case 'FNPBCDRMC23'

            lamParas =  [10 20 30 40 50 80 100];
            for iparas = 1 :  length(lamParas)
                lamdas = lamParas(iparas);

                tic;
                for ch=1:3
                    [L, err, iter, Times] = ReRMCrFig_FNPBCD(noisy_image(:,:,ch), ind, lamdas, 1e-5, 2/3, param);
                    disp(['ch_num = ' num2str(ch), ' iter = ' num2str(iter)]);
                    Xhat(:,:,ch) = L; 
                end
                time_cost = toc;
                Xhat = max(Xhat,0);
                Xhat = min(Xhat,255);
                ourpsnr = PSNR(Xfull,Xhat,max(Xfull(:)));
                disp([' lambda = ' num2str(lamdas), ' ValuePSNR = ' num2str(ourpsnr), ' iter= ' num2str(iter), ' time= ' num2str(time_cost)]);
                
            end



    end

end
 
 