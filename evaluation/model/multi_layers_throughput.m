% THROUGHPUT of two adjacent convolution layers depends on number of
% batches.This scripts evaluates to what extent can number of batches
% increase the throughput of the two layers design.

H = 32;
W = 32;
C = 512;
F = 512;
K = 3;
num_ops = @(N) (N * ((H - K + 1) * (W - K + 1) * C * F * K^2) * 2 * 2);
num_cycles = @(N) ((N + 1) * (H * W * F * C));
perf = @(N) (num_ops(N) * 10^(-9) / (num_cycles(N) / (100 * 10^6)));

X = 1:100;
Y = arrayfun(perf, X);
plot(X, Y)