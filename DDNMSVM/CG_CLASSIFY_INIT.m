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


function [f, df] = CG_CLASSIFY_INIT(VV,Dim,w2probs,target);
l1 = Dim(1);
l2 = Dim(2);
N = size(w2probs,1);
% Do decomversion.
  w_class = reshape(VV,l1+1,l2); %41*1
  w2probs = [w2probs  ones(N,1)];  % 284*41

%   targetout = exp(w2probs*w_class);
%   targetout = targetout./repmat(sum(targetout,2),1,10);
%   f = -sum(sum( target(:,1:end).*log(targetout))) ;
makebatches;
targetout=w2probs*w_class; % 284*1
AA=ones(size(bre_train,1),1)-target.*targetout;
AA=max(AA,0);

C=32;
A=(AA)'*AA;
% f=0.5*(w_class)'*w_class+C*(AA)'*AA;  % the objective function

f=0.5*(w_class)'*w_class+C*A;  % the objective function

IO=-2*C*target.*AA;
Ix_class=IO;
dw_class =w_class+(w2probs)'*Ix_class;  % 501*1
df = [dw_class(:)']';


% IO = (targetout-target(:,1:end));
% Ix_class=IO; 
% dw_class =  w3probs'*Ix_class; 
% 
% df = [dw_class(:)']'; 

