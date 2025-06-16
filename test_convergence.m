addpath('utils');

clear all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 64;                                       % patch size 

T0 = 8;                                       % sparsity level for each representation
T1=round((0.2)*(n^2));                        % sparsity level for the matrix B in the decomposition W=B*\Phi

numiter = 900;                                % Number of iterations for AM algorithm
cbb=floor(numiter/2);                         % number of iterations before starting the hard-thresholding operation for matrix B

W0 = kron(dctmtx(sqrt(n)), dctmtx(sqrt(n)));  % 2D DCT initialization, canonical transform factor
B0=W0;                                        % learnt factor initialization

lambda0 = 2.1e-6;                             % Bresler method parameter for log-determinant and Frobenius norm



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Loading and Preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load training set
barbara = struct2cell(load('data/barbara.mat')); barbara = barbara{1}; 
couple = struct2cell(load('data/couple.mat')); couple = couple{1}; 
lena = struct2cell(load('data/lena.mat')); lena = lena{1}; 

% vectorize
[blocks_barbara] = my_im2col(barbara, [sqrt(n), sqrt(n)], sqrt(n));
[blocks_couple] = my_im2col(couple, [sqrt(n), sqrt(n)], sqrt(n));
[blocks_lena] = my_im2col(lena, [sqrt(n), sqrt(n)], sqrt(n));

% concatenate
[blocks] = [blocks_barbara, blocks_couple, blocks_lena];

% subtract the means
br = mean(blocks);
TE = blocks - (ones(n, 1) * br);
YH = TE; 

% set the sparsity levels
STY = T0 * ones(1, size(YH, 2)); 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run Transforms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% Unstructured Bresler Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l2_bresler = lambda0 * (norm(YH, 'fro'))^2;
l3_bresler = l2_bresler;
[W_bresler, X_bresler, error_bresler, error2_bresler] = TLBresler(W0, YH, numiter, l2_bresler, l3_bresler, STY);
fprintf('BreslerTL Done\n');


%%%%%%%%%%%%%%%%%%%%%% Unstructured Conditioned Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% use W_bresler for setting the condition number and Frobenius norm for comparison
kappa = cond(W_bresler);        
tau = norm(W_bresler, 'fro');  
[W_cond, X_cond, error_cond, error2_cond, cond_nums_cond_tl] = TLConditioned(W0, YH, numiter, STY, kappa, tau);
fprintf('ConditionedTL Done\n');


%%%%%%%%%%%%%%%%%%%%%%% Structured Conditioned Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

YH2 = W0*YH;  % pre-processed training set is used by the learning function
[B_doubly_cond, X_doubly_cond, error_doubly_cond, error2_doubly_cond, cond_nums_doubly_cond] = ConditionedDoublySparse(B0, YH2 ,numiter, STY, kappa, tau, T1, cbb);
fprintf('DoublyCondTL Done\n\n');
kappa_B = cond(B_doubly_cond);
fprintf('Target kappa: %.4e, actual kappa: %.4e, abs diff: %.4e\n', kappa, kappa_B, abs(kappa - kappa_B));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DCT Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

YH2 = W0*YH;
[s]=sort(abs(YH2),'descend'); 
X = YH2.*(bsxfun(@ge,abs(YH2),s(STY))); 

error_dct = ones(1, numiter) * norm(X - YH2, 'fro');
error2_dct = ones(1, numiter) * (norm(X - YH2, 'fro') / norm(YH2, 'fro'));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

errors = {error_dct, error_bresler, error_cond, error_doubly_cond};
errors2 = {error2_dct, error2_bresler, error2_cond, error2_doubly_cond};
labels = {'Discrete Cosine Transform', 'Bresler', 'Unstructured Conditioned', 'Structured Conditioned'};

plot_convergence(numiter, errors, errors2, labels, kappa, tau, T0, 'test_convergence.png');
