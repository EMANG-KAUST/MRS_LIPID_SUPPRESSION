function [ yscsa, h] = SCSAden( yf,yf0,v )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
gm=0.5; fs=1;
%%
% yf=temp;
% f=1:length(temp);
 f=1:length(yf);
Denoising_coeff=0.000000000005;
width_peaks=5;          %  The ratio w.r.t metabolite amplitude to set the threshold from which the peak can be selected
Th_peaks_ratio=2;       %  The with of the peak from its center(max values)
y_p1=gradient(yf);
    y_p2=gradient(y_p1);
    curv=abs(y_p2)./(1+y_p1.^2).^(1.5);
  lambda=max(yf)/sum(curv)*10^v;
[yscsa, h]=SCSA_MRS_Denoising(yf, gm, fs, lambda);

if size(yf0,1)>1
    yf0=yf0';
    yf=yf';
    de=de';
end


end