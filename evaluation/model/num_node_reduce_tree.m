function [ result ] = num_node_reduce_tree( V )
%num_node_reduce_tree Get the number of nodes in a reduce tree.

if (V == 1)
    result = 0;
else
    N = V;
    result = 0;
    while (N > 1)
        M = floor(N / 2);
        N = N - M;
        result = M + result;
    end

end

