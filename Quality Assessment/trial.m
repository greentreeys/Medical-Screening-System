clc;
clear all;
close all;

im = imread('sample.jpg');
mask = imread('mask.jpg');
[feats loc] = filterderivative_withLoc(im, mask);
sampled_data = datasample(feats,150,1,'Replace',false);
for i = 1:150
    sampled_data(i,:) = (sampled_data(i,:)-mean(sampled_data(i,:)))/std(sampled_data(i,:));
end
idx = kmeans(sampled_data,5);
isc_counts = hist(idx,5)';
[rcounts,x] = imhist(double(im(:,:,1)),5);
[gcounts,x] = imhist(double(im(:,:,2)),5);
[bcounts,x] = imhist(double(im(:,:,3)),5);
features = [isc_counts' rcounts' gcounts' bcounts']
% plot(idx(:,1)==1,'.'); hold on;
% plot(idx(:,1)==2,'r.'); hold on;
% plot(idx(:,1)==3,'g.'); hold on;
% plot(idx(:,1)==4,'k.'); hold on;
% plot(idx(:,1)==5,'c.'); hold off;
% axis([min(loc(:,1))-10 max(loc(:,1))+10 min(loc(:,2))-10 max(loc(:,2))+10]);