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

if ~exist('scale')
    std=[1 2 4 8 16]; %% Number of scale
else
    std=scale;
end

[loc(:,1), loc(:,2)]= find(mask==1);
feats=zeros(size(loc,1),25);

im = extension(im,mask);% Fundus extension

[tmp num]= size(std);
epsilon=1e-2;

Lvv= zeros(size(im,1), size(im,2), size(std,2));
Lvw= zeros(size(im,1), size(im,2), size(std,2));
Lww= zeros(size(im,1), size(im,2), size(std,2));
pos= 1;
z = 1;
for k=1:num
    sigma=std(k);
    % To create Gaussian filter similiar to the one used in gaussgradient function
    halfsize=ceil(sigma*sqrt(-2*log(sqrt(2*pi)*sigma*epsilon)));
    sze=2*halfsize+1;
    gauss= fspecial('gaussian',sze,sigma);
    L(:,:,k)= imfilter(im,gauss);
    subplot(5,5,z), imshow(L(:,:,k),[]);
    z = z + 1;
    [Lx,Ly]=gaussgradient(im,sigma);
    [Lxx,Lxy]=gaussgradient(Lx,sigma);
    [Lyx,Lyy]=gaussgradient(Ly,sigma);
    
    Lw(:,:,k)= sqrt(Lx.^2+Ly.^2);
    subplot(5,5,z),imshow(Lw(:,:,k),[]);
    z = z + 1;
    % Computing numerator of Lvv
    Lvv_num= (- 2 * Lx .* Lxy .* Ly) + (Lxx .* (Ly.*Ly)) + ((Lx.*Lx).* Lyy);
    % Computing numerator of Lvw 
    Lvw_num= (-1*(Lx.*Lx) .* Lxy) + ((Ly.*Ly) .* Lxy) + (Lx.*Ly .*(Lxx-Lyy));
    % Computing numerator of Lww 
    Lww_num= ((Lx.*Lx) .* Lxx) + (2* Lx.* Lxy.*Ly) + ((Ly.*Ly) .* Lyy);
    % Denominator for the computation of filters Lvv, Lvw and Lww 
    den = (Lx.*Lx+ Ly.*Ly+0.001);
    subplot(5,5,z), imshow(Lvv_num./den, []);
    z = z + 1;
    subplot(5,5,z), imshow(Lvw_num./den, []);
    z = z +1;
    subplot(5,5,z), imshow(Lww_num./den, []);
    z = z + 1;
    
    for c=1:size(loc,1)
        i = loc(c,1); j = loc(c,2);
        feats(c,pos)=L(i,j,k);
        feats(c,pos+1)=Lw(i,j,k);
        % Computation for Lvv
        feats(c,pos+2) = Lvv_num(i,j)/den(i,j);
        % Computation for Lvw
        feats(c,pos+3) = Lvw_num(i,j)/den(i,j);
        % Computation for Lww
        feats(c,pos+4) = Lww_num(i,j)/den(i,j);
    end
    pos=pos+5;
end
