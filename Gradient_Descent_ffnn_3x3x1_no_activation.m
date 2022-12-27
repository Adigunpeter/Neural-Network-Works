clc;
clear all %clears memory
%Name: Peter Oluwasayo Adigun
%Class: CS 4730-5730, Artificial Neural Networks
%Date: 4-22-2021
%Title:Gradient Descent and 3x3x1 FFNN
%Notes: Version 1 of 1

%Create data 
n=10;
x_0 = ones(n,1);
x_1 = rand(n,1);
x_2 = rand(n,1);

X = [x_0 x_1 x_2]; %data matrix 

%labeled data, y_l
y_l= rand(n,1);

%Generate random W_0 and W_1 matrices
hidden = 3;
input = 3;
output = 1;

W_0 = rand(hidden,input);
W_1 = rand(output,hidden);

%Forward Propagation I=>H
H_0 = forward_propagation(W_0,X);

%Forward Propagation H=>O
y_0 = forward_propagation(W_1,H_0');

% Error e_i
e = y_l - y_0';

%Back propagation
%Calculate the cost function
J = 1./(2.*n).*(e'*e);

%Calculate dJ/dw_0_00
temp = e.*x_0;
dJ_dw_0_00 = -1./n.*sqrt(temp'*temp);

sum_0 =0;
for i=1:n
  sum_0 = sum_0 + temp(i);
endfor
dJ_dw_0_00 = -1./n.*sum_0;
