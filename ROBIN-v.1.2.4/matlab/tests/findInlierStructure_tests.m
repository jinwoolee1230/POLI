addpath(genpath('../build-matlab/matlab'))

adj_mat = [0, 1, 1, 1;
           1, 0, 1, 1;
           1, 1, 0, 1;
           1, 1, 1, 0];
       
[max_core_indices, time] = findInlierStructure_mex(adj_mat, 0);
assert(all(max_core_indices == [0, 1, 2, 3]))

[max_clique_indices, time] = findInlierStructure_mex(adj_mat, 0);
assert(all(max_clique_indices == [0, 1, 2, 3]))


adj_mat = [0, 1, 1, 0;
           1, 0, 1, 0;
           1, 1, 0, 0;
           0, 0, 0, 0];
       
[max_core_indices, time] = findInlierStructure_mex(adj_mat, 0);
assert(all(max_core_indices == [0, 1, 2]))

[max_clique_indices, time] = findInlierStructure_mex(adj_mat, 0);
assert(all(max_clique_indices == [0, 1, 2]))