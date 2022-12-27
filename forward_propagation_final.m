 clc
clear memory
%Name: Adigun Peter OLuwasayo
%Class: CS 4730-5730, ANN
%Date: 4-29-2021
%Title:Sigmoid Function
%Notes: Version 1 of 1

   %Begin Calculations
   %Define x = [1 0 1]', 3x1 and y_l = [1]', 1x1
   x = [1;0;1];
   y_l = [1];

    %Define weight matrices, W_0 and W_1 (5x5, 1x5)
    W_0 = rand(3,3);
    W_1 = rand(1,3);
    
    %Calculate (a) h_0 = W_0*x, 3x1 (3x3*3x1)
    v_0 = W_0*x % 
    %Sending summed values through sigmoid activation function for hidden layer0
    h_1 = Sigmoid(v_0)
    %Calculate (b) h_1 = W_1*h_0 1x1 (1x3*3x1)
    h_2 = W_1*h_1 % 
    %Sending summed values through sigmoid activation function for output layer
    y_p = Sigmoid(h_2)
    %Calculate e = (y_p - y_l), (1x1 - 1x1), 2x1
    e = y_p - y_l; % e = 1x1
    %Display results
    y_p
    e
    v_0

