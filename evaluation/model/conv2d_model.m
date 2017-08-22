function [ U ] = conv2d_model( P )
% Compute the resource usage of the conv2d block by its parameters

[N, M] = size(P);

U = zeros(N, 4);
for n = 1:N
    if P(n, 1) == 32
        Ua = [0 32 0 0];
        Um = [0 1 0 2];
    end
    [lut, ff, bram, dsp] = dotprod_model(Ua, Um, 9);
    
    U(n, 1) = lut * prod(P(n, 2:4));
    U(n, 2) = ff * prod(P(n, 2:4));
    U(n, 3) = bram * prod(P(n, 2:4));
    U(n, 4) = dsp * prod(P(n, 2:4));
end

end

