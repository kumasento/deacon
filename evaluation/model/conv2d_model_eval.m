filename = fullfile('..', 'data', 'Conv2DKernel.csv');
data = csvread(filename, 1);

result = conv2d_model(data(:, 1:4));
golden = data(:, 5:8);

disp(result);
disp(golden);

[N, M] = size(data);
error = sum(abs(result - golden) ./ golden, 1);
error = error / N;
disp(error);