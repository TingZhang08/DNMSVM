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


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

% clear all
% close all

num_of_running=10;
train=zeros(num_of_running,1);
test=zeros(num_of_running,1);

for ii=1:num_of_running

maxepoch=10; 
numhid=40; numpen=40; 
% numpen2=2000; 




makebatches;
[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm;
hidrecbiases=hidbiases; 
save mnistvhclassify vishid hidrecbiases visbiases;

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthpclassify hidpen penrecbiases hidgenbiases;

% fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
% batchdata=batchposhidprobs;
% numhid=numpen2;
% restart=1;
% rbm;
% hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
% save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;

backpropclassify; 


  train(ii,1)=train_correct(1,end)/size(bre_train,1);
  test(ii,1)=test_correct(1,end)/size(bre_test,1);
  mean(train)
  mean(test)


end

