clc;
clear all;
close all;

im = imread('sample.jpg'); %Reading image
mask = create_mask(im,10); %Creating mask
[feats loc] = filterderivative_withLoc(im, mask);