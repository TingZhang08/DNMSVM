% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


function [f, df] = CG_CLASSIFY(VV,Dim,XX,target);

l1 = Dim(1);
l2 = Dim(2);
% l3= Dim(3);
l4= Dim(3);
l5= Dim(4);
N = size(XX,1);

% Do decomversion.
 w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2 = reshape(VV(xxx+1:xxx+(l2+1)*l4),l2+1,l4);
 xxx = xxx+(l2+1)*l4;
%  w3 = reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
%  xxx = xxx+(l3+1)*l4;
 w_class = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);


  XX = [XX ones(N,1)];
  w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];

%   targetout = exp(w3probs*w_class);
%   targetout = targetout./repmat(sum(targetout,2),1,10);
%   f = -sum(sum( target(:,1:end).*log(targetout))) ;
% 
% IO = (targetout-target(:,1:end));
% Ix_class=IO; 
% dw_class =  w3probs'*Ix_class; 


targetout=w2probs*w_class;
AA=ones(size(XX,1),1)-target.*targetout;
for i=1:size(AA,1)
    if AA(i,1)<0
        AA(i,1)=0;
    end
end
C=32;
f=0.5*(w_class)'*w_class+C*(AA)'*AA;  % the objective function
IO=-2*C*target.*AA;
Ix_class=IO;
dw_class =w_class+(w2probs)'*Ix_class; % 501*1


% Ix3 = (Ix_class*w_class').*w3probs.*(1-w3probs);
% Ix3 = Ix3(:,1:end-1);
% dw3 =  w2probs'*Ix3;

Ix2 = (Ix_class*w_class').*w2probs.*(1-w2probs); 
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw_class(:)']'; 

% df = [dw1(:)' dw2(:)' dw3(:)' dw_class(:)']'; 
