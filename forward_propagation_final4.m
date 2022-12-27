clc;
clear all %clears memory
%Name: Adigun peter oluwasayo
%Class: CS 4730-5730, Artificial Neural Networks
%Date: 3-30-2021
%Title:Gradient Descent and Linear Regression
%Notes: Version 1 of 1

%Create data 
n=10;
x_0 = ones(n,1);
x_1 = rand(n,1);
x_2 = rand(n,1);

X = [x_0 x_1 x_2]'; %data matrix 

%labeled data, y_l
y_l= rand(n,1)';

%Generate random W_0 and W_1 matrices
hidden = 4;
hidden1 = 4;
input = 3;
output = 1;

W_0 = rand(hidden,input);
W_1 = rand(hidden1,hidden);
W_2 = rand(output,hidden1);
N=10;
for epochs=1:N
  %Forward Propagation I=>H
  H_0 = forward_propagation(W_0,X');

  %Forward Propagation H=>O
  H_1 = forward_propagation(W_1,H_0');
  
  y_0 = forward_propagation(W_2,H_1');

  %Error e_i
  e = y_l - y_0;
end
size(e)
%Back Propagation
  %Calculate the Cost function, J(W_0,W_1,y_l,y_0)
  J = 1./(2.*n).*(e*e');
  %Calculate dJ/dw_0_00, dJ/dw_0_01 ... dJ/dw_0_22
  
  for j=1:size(X,1)
    temp(:,j) = e'.*X(j,:)';
  end
  %Calculate Jacobian, J_0
  for k=1:size(H_0,1)
    for j=1:size(X,1)
      sum =0;
      for i=1:n
        sum = sum + temp(i,j);
      end
      dJ_dw_0(k,j) = -W_1(1,k)./n.*sum;
    end
  end
  Jacobian_0 = dJ_dw_0;
%Calculate dJ/dw_1_00, dJ/dw_1_01, and dJ/dw_1_02
sum_0(1) = W_0(1,:)*X(:,1);
sum_0(2) = W_0(2,:)*X(:,1);
sum_0(3) = W_0(3,:)*X(:,1);
sum_0(4) = W_0(4,:)*X(:,1);

  %Calculate dJ/dw_1_00, dJ/dw_1_01 ... dJ/dw_1_02
  for j=1:size(H_0,1)
    temp(:,j) = e.*sum_0(j);
  end
  %Calculate Jacobian, J_1
  for k=1:size(W_1,1)
    for j=1:size(H_0,1)
      sum =0;
      for i=1:n
        sum = sum + temp(i,j);
      end
      dJ_dw_1(k,j) = -1.0./n.*sum;
    end
  end
  Jacobian_1 = dJ_dw_1;
  %Calculate dJ/dw_1_00, dJ/dw_1_01 ... dJ/dw_1_02
sum_1(1) = W_1(1,:)*H_0(:,1);
sum_1(2) = W_1(2,:)*H_0(:,1);
sum_1(3) = W_1(3,:)*H_0(:,1);
sum_1(4) = W_1(4,:)*H_0(:,1);

   for j=1:size(H_1,1)
    temp(:,j) = e.*sum_1(j);
  end
  %Calculate Jacobian, J_1
  for k=1:size(W_2,1)
    for j=1:size(H_1,1)
      sum =1;
      for i=1:n
        sum = sum + temp(i,j);
      end
      dJ_dw_2(k,j) = -1.0./n.*sum;
    end
  end
  Jacobian_2 = dJ_dw_2;

  %Calculate Update Law for Delta_W_0
  learning_rate_1 = 0.01;
  W_0_new = W_0 -learning_rate_1*Jacobian_0;
  W_1_new = W_1 -learning_rate_1*Jacobian_1;
  W_2_new = W_2 -learning_rate_1*Jacobian_2;
  %Calculate change in W_0 and W_1
  Delta_W_0 = W_0_new - W_0;
  Delta_W_1 = W_1_new - W_1;
  Delta_W_2 = W_2_new - W_2;
  %Replace old values with new values for W_0 and W_1
  W_0 = W_0_new;
  W_1 = W_1_new;
  W_2 = W_2_new;

%Output the change in W_0 and W_1 as well as W_0 and W_1
Delta_W_0
Delta_W_1
Delta_W_2


W_0
W_1
W_2
%Output final Jacobians, J_0 and J_1
Jacobian_0
Jacobian_1
Jacobian_2

%Construct the y_p = f(W_0,W_1,W_2,X) model
H_p0 = W_0*X;
H_p1 = W_1*H_p0;
y_p = W_2*H_p1;

plot(e)




