a=imread('/home/lasan/study/CVIT/Sem8/vessel/pink/1265753352183.JPEG');
b=imread('/home/lasan/study/CVIT/Sem8/vessel/pink_vss//1265753352183.JPEG');
c=im2bw(b,0.01);
d=logical(1-c); %groundtruth

% a in green channel so converting to rgb
e=zeros(size(a,1),size(a,2),3);
e(:,:,2)=a;
e=double(e) ./ 255;

%% Symmetric filter params
symmfilter = struct();
symmfilter.sigma     = 2.4;
symmfilter.len       = 8;
symmfilter.sigma0    = 3;
symmfilter.alpha     = 0.7;

%% Asymmetric filter params
asymmfilter = struct();
asymmfilter.sigma     = 1.8;
asymmfilter.len       = 22;
asymmfilter.sigma0    = 2;
asymmfilter.alpha     = 0.1;

max=0;

%for i=1:5
%    symmfilter.sigma=i
    for j=1:5:41
        symmfilter.len=j;
        for k=0.1:0.1:0.5
            for l= 10:5:40
                [resp segresp r1 r2] = BCOSFIRE(e, symmfilter, asymmfilter, k, l);
                f=xor(segresp,d);
                f=logical(1-f);
                count=sum(f(:));
                if count > max
                    max = count
                    w=i;
                    x=j;
                    y=k;
                    z=l;
                end
            end
        end
    end
%end