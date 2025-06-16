addpath('utils');

clear all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 64;                                      % patch size 

T0 = 8;                                      % sparsity level for each representation
T1=round((0.20)*(n^2));                      % sparsity level for matrix B in the decomposition W=B*\Phi

numiter = 900;                               % number of iterations for AM algorithm
cbb=floor(numiter/3);                        % number of iterations before starting the hard-thresholding operation for the matrix B

W0 = kron(dctmtx(sqrt(n)), dctmtx(sqrt(n))); % 2D DCT initialization, canonical transform factor
B0=W0;                                       % learnt factor initialization



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Loading and Preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load training set
barbara = struct2cell(load('data/barbara.mat')); barbara = barbara{1}; 
couple = struct2cell(load('data/couple.mat')); couple = couple{1}; 
lena = struct2cell(load('data/lena.mat')); lena = lena{1}; 

% vectorize
[blocks_barbara] = my_im2col(barbara, [sqrt(n), sqrt(n)], sqrt(n));
[blocks_couple] = my_im2col(couple, [sqrt(n), sqrt(n)], sqrt(n));
[blocks_lena] = my_im2col(lena, [sqrt(n), sqrt(n)], sqrt(n));

% concatenate training data
[blocks] = [blocks_barbara, blocks_couple];

% subtract the means
br = mean(blocks);
TE = blocks - (ones(n, 1) * br);
YH = TE; 

% set the sparsity levels
STY = T0 * ones(1, size(YH, 2)); 

% prepare lena test
br_lena = mean(blocks_lena);
TE_lena = blocks_lena - (ones(n, 1) * br_lena);
YH_lena = TE_lena; 

YH_lena = W0*YH_lena;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run Transforms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Small Kappa %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kappa_small = 1.56;
tau = 4.87;  
YH2 = W0*YH;  
[B_small_kappa, ~, ~, ~, ~, last_x_small_kappa] = ConditionedDoublySparse(B0, YH2 ,numiter, STY, kappa_small, tau, T1, cbb);
kappa_small = cond(B_small_kappa);

fprintf('Small Kappa Test Done - 1/3\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Good Kappa %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kappa_good = 8.23;        
tau = 3.62;  
YH2 = W0*YH; 
[B_good_kappa, ~, ~, ~, ~, last_x_good_kappa] = ConditionedDoublySparse(B0, YH2 ,numiter, STY, kappa_good, tau, T1, cbb);
kappa_good = cond(B_good_kappa);        

fprintf('Good Kappa Test Done - 2/3\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Big Kappa %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kappa_big = 1011.22;        
tau = 2.33;  
YH2 = W0*YH;
[B_big_kappa, ~, ~, ~, ~, last_x_big_kappa] = ConditionedDoublySparse(B0, YH2 ,numiter, STY, kappa_big, tau, T1, cbb);
kappa_big = cond(B_big_kappa);        

fprintf('Big Kappa Test Done - 3/3\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% test lena
lena_small_kappa = B_small_kappa * YH_lena;
lena_good_kappa = B_good_kappa * YH_lena;
lena_big_kappa = B_big_kappa * YH_lena;

kappas = [kappa_small, kappa_good, kappa_big];
train = {last_x_small_kappa, last_x_good_kappa, last_x_big_kappa};
test  = {lena_small_kappa, lena_good_kappa, lena_big_kappa};

plot_magnitude(train, test, kappas, T0, 'test_magnitude.png');
