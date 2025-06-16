function [B, XB, error, error2, cond_nums, last_x] = ConditionedDoublySparse(B,Y,numiter, STY, kappa, targetWnorm, T1, cbb);
addpath('conditionedTLroutines');
rng(0);

% Implementation based on the paper "Learning Explicitly Conditioned Transforms"
% by A. Pătrașcu, C. Rusu, and P. Irofti, and adapted within the transform learning framework
% from the series of papers by S. Ravishankar and Y. Bresler.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[K,n]=size(B); 
XB=zeros(K,size(Y,2)); 

ix=find(STY>0);
q=Y(:,ix);
STY=STY(:,ix); 
N=size(q,2); 

ez=K*(0:(N-1)); 
STY=STY + ez; 
Y = Y(:, ix);

error = []; 
error2 = [];
kappas = []; 
cond_nums = []; 
last_x = []; 

for i=1:numiter

    X1=B*q; 
    last_x = X1; 
    [s]=sort(abs(X1),'descend'); 

    X = X1.*(bsxfun(@ge,abs(X1),s(STY))); 
    
    if isempty(kappas)
        kappas = linspace(numiter*kappa, kappa, numiter); 
        
        if kappa <= 2
            B = Y*X';
            [U, S, V] = svd(B); 

        else
            B = (Y'\X')';

            cll=1e-1;
            while(abs(det(B)) <= 10^(-250))
                B = B + ((rand(K,n) - 0.5)*cll);
            end

            [U, S, V] = svd(B);
        end
        
        the_sum = sum(diag(S)); 
        l_min = min(diag(S)); 
        
        [d_cvx, V] = get_spectrum_from_data(U, V, X, Y, kappa, the_sum, l_min);
    end
    
    %%% get the U
    [Q, ~, T] = svd(diag(d_cvx)*V'*Y*X');
    U = (Q*T')';
    
    %%% get the V
    [Uu, Ss, Vv] = svd((U'*X)*Y');
    V = (Uu*Vv')';
    
    %%% get the spectrum
    [d_cvx, V] = get_spectrum_from_data(U, V, X, Y, kappa, the_sum, l_min);
    
    B = U*diag(d_cvx)*V';

    % post-thresholding
    if(i>cbb) % done only after initial `cbb' number of iterations
        [~,vu]=sort(abs(B(:)),'descend');
        Bg=zeros(size(B));
        Bg(vu(1:T1))=B(vu(1:T1));
        B=Bg;
    end

    B = B/norm(B, 'fro')*targetWnorm;

    cond_nums = [cond_nums cond(B)];
    error = [error norm(X - B*Y, 'fro')];
    error2 = [error2 norm(X - B*Y, 'fro')/norm(B*Y, 'fro')];
    if (i == 84)
        stop = 1;
    end
end
XB(:,ix)=X;
