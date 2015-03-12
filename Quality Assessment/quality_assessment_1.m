% Input (im): Any image
% Output (features): The feature vector obtained from histograms of ISC
% clustering and RGB histograms
% Displays the image after clustering. Currently the code clusters all the pixel in order to display the
% complete image of the eye. If this is not required comment code from
% lines 23 to 49 and uncomment 22
clc;
clear all;
close all;

im = imread('image034.png'); %Reading image
mask = create_mask(im,10); %Creating mask

figure,
[feats loc] = filterderivative_withLoc(im, mask); %Performing filterbank to obtain features

sampled_data = datasample(feats,150,1,'Replace',false); %150 vectors are obtained by random sampling
%Converting the vectors to zero mean and unit variance
sampled_data = zscore(sampled_data');
sampled_data = sampled_data';

%Applying kmeans to cluster the feature vectors into 5 classes
% idx = kmeans(sampled_data,5); 
idx = kmeans(feats,5);
for i = 1:size(loc,1)
     if idx(i) == 1
         im(loc(i,1),loc(i,2),:) = 0;
     elseif idx(i) == 2
        im(loc(i,1),loc(i,2),1) = 0;
        im(loc(i,1),loc(i,2),2) = 0;
        im(loc(i,1),loc(i,2),3) = 255;
    elseif idx(i) == 3
        im(loc(i,1),loc(i,2),1) = 0;
        im(loc(i,1),loc(i,2),2) = 255;
        im(loc(i,1),loc(i,2),3) = 0;
    elseif idx(i) == 4
        im(loc(i,1),loc(i,2),1) = 255;
        im(loc(i,1),loc(i,2),2) = 0;
        im(loc(i,1),loc(i,2),3) = 0;
    elseif idx(i) == 5
        im(loc(i,1),loc(i,2),1) = 255;
        im(loc(i,1),loc(i,2),2) = 255;
        im(loc(i,1),loc(i,2),3) = 255;
    end
end
figure, imshow(im);
isc_counts = hist(idx,5); %Features from ISC histogram
% Features from RGB histograms
[rcounts,x] = imhist(double(im(:,:,1)),5);
[gcounts,x] = imhist(double(im(:,:,2)),5);
[bcounts,x] = imhist(double(im(:,:,3)),5);
%Feature vector
features = [isc_counts rcounts' gcounts' bcounts']
