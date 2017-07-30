% Evaluate dotprod model

filename = fullfile('..', 'data', 'dotprod.csv');
data = csvread(filename, 1);
data32b = data(1:5, :);
Ua32b = [0 32 0 0];
Um32b = [0 1 0 2];
V = data32b(:, 2);
[lut, ff, bram, dsp] = dotprod_model(Ua32b, Um32b, V);
g_lut = data32b(:, 3);
g_ff = data32b(:, 4);
g_bram = data32b(:, 5);
g_dsp = data32b(:, 6);

% Two result matrices
U = [lut ff bram dsp];
g_U = [g_lut g_ff g_bram g_dsp];

% Check lut usage
fitlut = fit( (2*V-1), g_lut, 'poly1', 'Robust', 'on' );

subplot(1, 2, 1);
plot( V, fitlut.p1 * (2*V-1) + fitlut.p2 );
hold on;
scatter( V, g_lut, 'o');
xlabel('Vector Size');
ylabel('LUT Usage');
hold off;

% Check ff usage
fitff = fit((2*V-1), g_ff - ff, 'poly1', 'Robust', 'on');

subplot(1, 2, 2);
plot( V, fitff.p1 * (2*V-1) + fitff.p2 );
hold on;
scatter( V, g_ff - ff, 'o');
xlabel('Vector Size');
ylabel('FF Usage');
hold off;

disp(fitlut);
disp(fitff);