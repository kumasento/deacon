function [ lut, ff, bram, dsp ] = dotprod_model( Ua, Um, V )
%dotprod_model Analysis model for Dot-Product
% Ua - adder resource usage
% Um - multiplier resource usage
% V - vector size

A = [ V V-1 ];
B = [ Um; Ua ];
U = A * B;

lut = U(:, 1);
ff = U(:, 2);
bram = U(:, 3);
dsp = U(:, 4);

end

