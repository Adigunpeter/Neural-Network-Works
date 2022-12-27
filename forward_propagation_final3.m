 clc
clear memory
%Name: Adigun Peter OLuwasayo
%Class: CS 4730-5730, ANN
%Date: 3-9-2021
%Title:Sigmoid Function
%Notes: Version 1 of 1

   %Begin Calculations
   %Define x = [ 0  1 0 1 ]', 4x1 and y_l = [1 0]', 2x1
   x = [1;0;1];
   y_l = [1];

    %Define weight matrices, W_0 and W_1 (5x5, 1x5)
    W_0 = rand(4,3);
    W_1 = rand(4,4);
    W_2 = rand(4,4);
    W_3 = rand(1,4);
    
    %Calculate (a) h_0 = W_0*x, 4x1 (4x3*3x1)
    v_0 = W_0*x % 
    %Sending summed values through sigmoid activation function for hidden layer0
    h_0 = Sigmoid(v_0)
    %Calculate (b) h_1 = W_1*h_0 4x1 (4x4*4x1)
    v_1 = W_1*h_0; % 
    %Sending summed values through sigmoid activation function for hidden layer0
    h_1 = Sigmoid(v_1)
    %Calculate (b) h_2 = W_2*h_1 4x1 (4x4*4x1)
    v_2 = W_2*h_1; % 
    %Sending summed values through sigmoid activation function for hidden layer1
    h_2 = Sigmoid(v_2)
    %Calculate (c) y_p = W_3*h_2 1x1 (1x4*4x1)
    h_3 = W_3*h_2 % 
    %Sending summed values through sigmoid activation function for output layer
    y_p = Sigmoid(h_3)
    %Calculate e = (y_p - y_l), (1x1 - 1x1), 1x1
    e = y_p - y_l; % e = 1x1
    %Display results
    y_p
    e
    v_0 
    v_1 
    v_2

