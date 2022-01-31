%% MRS signal Denosiing  using SCSA Method :
% This function denoised the real part of the MRS signal using and optimal
% value of the h the ensures the preservation of the metabolite area,
% determined by <Metabolite>, while remove as maximuim as possible the noise from the
% region determined by <Noise>

%% ######################  PARAMETERS ######################################
% Input
% ppm         : The MRS spectrum fequancies in ppm
% yf          : Noisy real art of the  complex MRS FID signal
% Th_peaks_ratio  : The ratio w.r.t metabolite amplitude to set the threshold from which the peak can be selected
% width_peaks  : The with of the peak from its center(max values)
% gm, fs  : SCSA parameter: gm=0.5, fs=1

% Output
% yscsa: Densoied real art of the  complex MRS FID signal
% h_op : The  real art of the  complex MRS FID signal
% Nh   : Densoied real art of the  complex MRS FID signal

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
%  Adviser:
%  Taous-Meriem Laleg (taousmeriem.laleg@kaust.edu.sa)
% Done: June,  2018
% King Abdullah University of Sciences and Technology (KAUST)

function [yscsa, h_op]=SCSA_MRS_Denoising(yf, gm, fs,lambda)


global shif
truth=[];
hhh=[];

%% Generate signals
fprintf('\n_____________________________________________________________')
fprintf('\n_____________________________________________________________\n\n')

%% MRS signal
y = real(yf);%/max(real(yf))*76;

% you can choose either a vecotr for h or one value
hh=max(y)/100 : 1 :max(y);  % search in this interval

%% Some specific values for the real signal

fprintf('\n-->Searching for the optimal h')
harray=[];
se=[];

% start the loop for several values of h

for i=1:length(hh)
    
    h = hh(i);%0.045;%15.2;
    [h, yscsa]= SCSA_1D(y,fs,h,gm);    
    y_res=y-yscsa;
    lenSCSA=length(yscsa);
    y_p1=gradient(yscsa);
    y_p2=gradient(y_p1);
    curv=abs(y_p2)./(1+y_p1.^2).^(1.5);
    q = simpsons(y_p2,1,lenSCSA);
    second_term=sum(curv);
    second_term=second_term*lambda;
    se(end+1)=second_term;

Cost_peak=0;

     if size(y_res,1)>1
         y_res=y_res';
     end
     
    Cost_peak=y_res*y_res';

  Cost_peak;

   Cost_function(i)=Cost_peak;
   Cost_function(i)=second_term+Cost_peak;
    fprintf('.')
    harray(end+1)=second_term+Cost_peak;
    res=Cost_peak;
    truth(end+1)=res;
    hhh(end+1)=h;
end

[M,I]=min(truth);
hhh(I);
h_op=min(hh(find(Cost_function==min(Cost_function))));

[h_op, yscsa]= SCSA_1D(y,fs,h_op,gm);

fprintf('--> MRS denoising is completed h=%f!!',h_op)

function [h, yscsa]= SCSA_1D(y,fs,h,gm)
Lcl = (1/(2*sqrt(pi)))*(gamma(gm+1)/gamma(gm+(3/2)));
N=max(size(y));
%% remove the negative part
Ymin=min(y);
y_scsa = y -Ymin;
%% Build Delta metrix for the SC_hSA
feh = 2*pi/N;
D=delta(N,fs,feh);
%% start the SC_hSA
Y = diag(y_scsa);
SC_h = -h*h*D-Y; % The Schrodinger operaor
% = = = = = = Begin : The eigenvalues and eigenfunctions
[psi,lamda] = eig(SC_h); % All eigenvalues and associated eigenfunction of the schrodinger operator
% Choosing  eigenvalues
All_lamda = diag(lamda);
ind = find(All_lamda<0);
%  negative eigenvalues
Neg_lamda = All_lamda(ind);
kappa = diag((abs(Neg_lamda)).^gm);
Nh = size(kappa,1); %%#ok<NASGU> % number of negative eigenvalues
if Nh~=0    
    % Associated eigenfunction and normalization
    psin = psi(:,ind(:,1)); % The associated eigenfunction of the negarive eigenvalues
    I = simp(psin.^2,fs); % Normalization of the eigenfunction
    psinnor = psin./sqrt(I);  % The L^2 normalized eigenfunction
    yscsa1 =((h/Lcl)*sum((psinnor.^2)*kappa,2)).^(2/(1+2*gm));
else
    
    psinnor = 0*psi;  % The L^2 normalized eigenfunction
    
    yscsa1=0*y;
    yscsa1=yscsa1-10*abs(max(y));
    disp('There are no negative eigenvalues. Please change the SCSA parameters: h, gm ')
end
if size(y_scsa) ~= size(yscsa1)
    yscsa1 = yscsa1';
end

%% add the removed negative part
yscsa = yscsa1 + Ymin;

squaredEIGF0=(h/Lcl)*(psinnor.^2)*kappa;


%**********************************************************************
%*********              Numerical integration                 *********
%**********************************************************************

% Author: Taous Meriem Laleg

function y = simp(f,dx);
%  This function returns the numerical integration of a function f
%  using the Simpson method

n=length(f);
I(1)=1/3*f(1)*dx;
I(2)=1/3*(f(1)+f(2))*dx;

for i=3:n
    if(mod(i,2)==0)
        I(i)=I(i-1)+(1/3*f(i)+1/3*f(i-1))*dx;
    else
        I(i)=I(i-1)+(1/3*f(i)+f(i-1))*dx;
    end
end
y=I(n);


%**********************************************************************
%*********             Delata Metrix discretization           *********
%**********************************************************************

%Author: Zineb Kaisserli

function [Dx]=delta(n,fex,feh)
ex = kron([(n-1):-1:1],ones(n,1));
if mod(n,2)==0
    dx = -pi^2/(3*feh^2)-(1/6)*ones(n,1);
    test_bx = -(-1).^ex*(0.5)./(sin(ex*feh*0.5).^2);
    test_tx =  -(-1).^(-ex)*(0.5)./(sin((-ex)*feh*0.5).^2);
else
    dx = -pi^2/(3*feh^2)-(1/12)*ones(n,1);
    test_bx = -0.5*((-1).^ex).*cot(ex*feh*0.5)./(sin(ex*feh*0.5));
    test_tx = -0.5*((-1).^(-ex)).*cot((-ex)*feh*0.5)./(sin((-ex)*feh*0.5));
end
Ex = full(spdiags([test_bx dx test_tx],[-(n-1):0 (n-1):-1:1],n,n));

Dx=(feh/fex)^2*Ex;