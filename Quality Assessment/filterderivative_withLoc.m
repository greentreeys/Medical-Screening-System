function [feats loc]= filterderivative_withLoc(im, mask,scale)
% [L Lw Lvv Lvw Lww]= filterderivative(im,scale)
% im= input single channel image
% Scale at which derivation computation required: default: [1 2 4 8 16];
% L Lw Lvv Lvw Lww : return derivative vector: dim == as per the give
% scales
% feats= 25 vector per pixel only belongs to the disk region.
% loc= contained respective location of the pixel which can be later used to display assigned cluster
% [r,g,b]hist= are the histogram of OD pixels with total of 25 bins (10 bins are mapped to one[255-> 25bins])


[m n dim]=size(im);
if dim>1
    im=double(im(:,:,2));
    disp 'filterderivative: <Extracted Green Channel from the color image >  '
else
    im=double(im);
end
green=im;
% disp ('Curvature space Lvv is computed.. ')

if ~exist('scale')
    std=[1 2 4 8 16]; %% Number of scale
    % std=[4]; %% Number of scale
else
    std=scale;
end

[xx, yy]= find(mask==1);
feats=zeros(size(xx,1),25);

[tmp num]= size(std);
epsilon=1e-2;

Lvv= zeros(size(im,1), size(im,2), size(std,2));
Lvw= zeros(size(im,1), size(im,2), size(std,2));
Lww= zeros(size(im,1), size(im,2), size(std,2));
pos= 1;
for k=1:num
    sigma=std(k);
    %     %% To create Gaussian filter similiar to the one used in gaussgradient
    %     %% function
    halfsize=ceil(sigma*sqrt(-2*log(sqrt(2*pi)*sigma*epsilon)));
    sze=2*halfsize+1;
    gauss= fspecial('gaussian',sze,sigma);
    L(:,:,k)= imfilter(im,gauss);
    
    [Lx,Ly]=gaussgradient(im,sigma);
    [Lxx,Lxy]=gaussgradient(Lx,sigma);
    [Lyx,Lyy]=gaussgradient(Ly,sigma);
    
    Lw(:,:,k)= sqrt(Lx.^2+Ly.^2);
%     size(xx)
    for c=1:size(xx,1)
        i= xx(c); j=yy(c);
        feats(c,pos)=L(i,j,k);
        feats(c,pos+1)=Lw(i,j,k);
        % Computation for Lvv
        factor1= - 2 * Lx(i,j) * Lxy(i,j) * Ly(i,j) + Lxx(i,j) * (Ly(i,j)*Ly(i,j));
        factor2=  ((Lx(i,j)*Lx(i,j))* Lyy(i,j));
        feats(c,pos+2)=  (factor1+factor2)/(Lx(i,j)*Lx(i,j)+ Ly(i,j)*Ly(i,j)+0.001);
        % Computation for Lvw
        factor1= - Lx(i,j)^2 * Lxy(i,j) + Ly(i,j)^2 * Lxy(i,j) + Lx(i,j)*Ly(i,j)*(Lxx(i,j)-Lyy(i,j));
        feats(c,pos+3)=  factor1/((Lx(i,j)^2+ Ly(i,j)^2)+0.001);
        %%% Computation for Lww
        factor1= Lx(i,j)^2 * Lxx(i,j) + 2* Lx(i,j)* Lxy(i,j)*Ly(i,j) + Ly(i,j)^2 * Lyy(i,j);
        feats(c,pos+4)=  factor1/((Lx(i,j)^2+ Ly(i,j)^2)+0.001);
        
    end
    pos=pos+5;
    %     disp (strcat('scale ',num2str(k),' is completed...'));
end
loc=[]; 
for c=1:size(xx,1)
    i= xx(c); j=yy(c);
    loc(c,1)= i; loc(c,2)= j;
end

