function [ lut, ff, bram, dsp ] = dotprod_model( Ua, Um, V )
%dotprod_model Analysis model for Dot-Product
% Ua - adder resource usage
% Um - multiplier resource usage
% V - vector size

A = [ V V-1 ];
B = [ Um; Ua ];
U = A * B;

lut = U(:, 1);
lut = lut + 28 * (2 * V - 1) + 30;
ff = U(:, 2);
ff = ff + 9 * (2 * V - 1) + 43;
bram = U(:, 3);
dsp = U(:, 4);

end

