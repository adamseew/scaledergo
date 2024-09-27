
clear;clc

% define bounds, adjust workspacwe
xb=[0 80];
yb=[0 80];

xlim_=[min([xb(1),xb(2),yb(1),yb(2)]),max([xb(1),xb(2),yb(1),yb(2)])];
ulim=[0 .6];

trees=load('trees.mat','trees');
trees=trees.trees;
treeSize=.05;

debug.trees=trees;

% transform workspace so that it has period 2
C=xlim_(1);
A=sum(abs(xlim_));

args.xlim_=xlim_;
args.A=A;
args.C=C;

% definitions for Mezic's controller
D=2; % dimension, e.g., 2D, 3D, etc.
L=2; % period

N=50000; % simulation horizon
M=8; % optimization horizon

SigmaG=[0.00101 1E-6;...
        1E-6 0.00101]; % cov. matrix, initial guess
MuG=[.5;.5]; % center, initial guess

no_alpha=5;
alpha=-ones(1,no_alpha)/no_alpha; % mixing coefficient
k=9; % number of freq. in Fourier transform
[K(1,:,:) K(2,:,:)]=ndgrid(0:1:k,0:1:k); % set of indices
dt=.01;

debug.xlim_=xlim_;
debug.x=nan(2,N);
debug.u=nan(2,N-1); % debug variables

args.L=L; % wrapping arguments for AUX functions
args.D=D;
args.alpha=alpha;
args.K=K;
history=12.5*M;
args.solution=[];
args.Gaussians=[];

CASADI_PATH='~/casadi'; % change to CasADi path
addpath(CASADI_PATH); % include CasADi
addpath('matlab'); % include functions for ergodic controller

import casadi.*;

% find an initial guess for the center of the gaussian
x=[.5;.5];
debug.x0=x;
debug.x(:,1)=x;

eps_alpha=.01;
eps_mu=.0025;
dist_trees=.5;

for i=1:length(alpha)
    args.Mu(:,i)=x;
    args.Sigma(:,:,i)=SigmaG;
end

for t=2:M-1:N

    opti=casadi.Opti();

    args.t=t;
    
    if (t<=history)
        T=1;
    else
        T=t-history;
    end
    PHI_K_VAL=opti.variable(length(K)^D,1); % aux variables

    X=opti.variable(2,M); % state
    U=opti.variable(2,M-1); % input
    ALPHA=opti.variable(length(args.alpha),1); % control variable
    MU=opti.variable(2,length(args.alpha));
    
    opti.set_initial(ALPHA,args.alpha); % setting initial guesses
    opti.subject_to(sum(ALPHA)<=1);
    opti.subject_to(ALPHA>0);
    
    args.Mu_=args.Mu;
    args.alpha_=args.alpha;

    opti.subject_to(0.01<=X<=.99);
    opti.subject_to(0<=MU<=1);
    opti.subject_to(0<=U<=.6);

    opti.subject_to(X(:,1)==debug.x(:,t-1));
    args.alpha=ALPHA;
    args.Mu=casadi.MX(2,length(args.alpha));
    args.Mu=MU;
    
    wt=zeros(length(K)^D,1);
    Lambda_k=zeros(length(K)^D);
    
    JX=(1/length(X))*sqrt((X(1,1)-x(1))^2+(X(2,1)-x(2))^2)^2;
    JU=0; % cost functions
    
    x=debug.x(:,t-1);
    for idx=1:M-1

        f_k_x_val=[];
        df_k_x_val=[];

        for k=1:length(K)

            f_k_x_val=[f_k_x_val;f_k_x(K(:,:,k),x,args)];
            df_k_x_val=[df_k_x_val df_k_x(K(:,:,k),x,args)*L^D];

            if idx == 1
                % phi_k is recomputed because it the underlying Sigma, Mu 
                % may change
                PHI_K_VAL((k-1)*length(K)+1:k*length(K))=...
                    -1*phi_k(K(:,:,k),args);
                  % !
                for j=1:length(args.K)
                    Lambda_k((k-1)*length(K)+j,(k-1)*length(K)+j)=...
                        (sum(args.K(:,j,k).^2)+1).^(-(args.D+1)/2);        
                end
            end
        end
        wt=wt+f_k_x_val;
        utilde=-1*df_k_x_val*Lambda_k*(wt/t-PHI_K_VAL);
        u=utilde*max(ulim)/(norm(utilde)+1E-1);
        x=x+u*dt;
        
        JU=JU+(1/length(U))*...
           sqrt((U(1,idx  )-u(1))^2+(U(2,idx  )-u(2))^2)^2; 
        JX=JX+(1/length(X))*...
           sqrt((X(1,idx+1)-x(1))^2+(X(2,idx+1)-x(2))^2)^2; 
        opti.subject_to(X(:,idx+1)==x);

        [nearest_trees,~]=rangesearch(trees,[.5,.5],dist_trees);
        for tree=trees(nearest_trees{1,1},:)' % tree constraint
            opti.subject_to((X(1,idx+1)-tree(1))^2+...
                            (X(2,idx+1)-tree(2))^2-treeSize^2>=0);
        end
    end

    JLOG=-1*log(sqrt((X(1,end)-debug.x(1,t-1))^2+(X(2,end)-debug.x(2,t-1))^2));     
    JMU=0;
    for idx=1:length(args.alpha)
        JMU=JMU+(1/length(args.alpha))*...
           sqrt((mean(debug.x(1,T:t-1)')'-MU(1,idx))^2+...
                (mean(debug.x(2,T:t-1)')'-MU(2,idx))^2)^2;
        opti.subject_to(...
            sqrt((args.Mu_(1,idx)-MU(1,idx))^2+...
                 (args.Mu_(2,idx)-MU(2,idx))^2)^2<=eps_mu); % not too distant from old estimate
        opti.subject_to(...
             sqrt(args.alpha_(idx)-ALPHA(idx))^2<=eps_alpha);
    end

    opti.minimize((JU+JX+JMU+JLOG));

    p_opts=struct('expand',false);  
    % s_opts=struct('max_iter',2500);
    opti.solver('ipopt',p_opts);%,s_opts);

    try
        sol=opti.solve();
        args.alpha=sol.value(ALPHA);
        MU=sol.value(MU);
        U=sol.value(U);
        X=sol.value(X);
        debug.phi_k_val=sol.value(PHI_K_VAL);
        args.solution{end+1}=sol.stats();
    catch exception
        args.alpha=opti.debug.value(ALPHA);
        MU=opti.debug.value(MU);
        U=opti.debug.value(U);
        X=opti.debug.value(X);
        debug.phi_k_val=opti.debug.value(PHI_K_VAL);
        args.solution{end+1}=exception.message;
    end

    args.Mu=nan(2,length(args.alpha));
    
    debug.u(:,t-1:t+M-3)=U(:,1:M-1);
    debug.x(:,t:t+M-2)=X(:,2:M);
    x=debug.x(:,t);

    args.Sigma(:,:,1)=cov(debug.x(:,T:t)');

    for i=1:length(alpha)
        args.Mu(:,i)=MU(:,i);
        args.Sigma(:,:,i)=args.Sigma(:,:,1);
    end
    
    args.Gaussians{end+1}={args.Mu,args.Sigma,args.alpha};

    clear 'PHI_K_VAL';

    plot_data(args, debug);
    drawnow

    if or(t==2,mod(t,100)==0)
        save('DATA.mat');
        axis=gca;
        exportgraphics(axis,'FIG.png');
    end


end

plot_data(args, debug);
drawnow




%%

function h=plot_data(args, debug)

    sampl=linspace(0,1,length(args.K)^args.D); % build samples per each 
                                               % point in space
    hold off;
    [probx,proby]=ndgrid(sampl,sampl);
    probx=probx(:);
    proby=proby(:);
    f_k_val_prob=nan((length(args.K)^args.D)^2,1);
    f_k_val_prob_val=nan(length(args.K)^args.D,1);

    for j=1:(length(args.K)^args.D)^2
        for k=1:length(args.K)
            f_k_val_prob_val((k-1)*length(args.K)+1:k*length(args.K))=...
                f_k_x(args.K(:,:,k),[probx(j);proby(j)],args);
        end        
        f_k_val_prob(j)=f_k_val_prob_val'*debug.phi_k_val;
    end

    h=surf(...
       reshape(probx,length(args.K)^args.D,length(args.K)^args.D),...
       reshape(proby,length(args.K)^args.D,length(args.K)^args.D),...
       reshape(f_k_val_prob,length(args.K)^args.D,length(args.K)^args.D),...
       'FaceColor','interp','EdgeAlpha',0,'FaceAlpha',.7...
    );
    colormap(flipud(bone(35)));
    z=get(h,'ZData');
    set(h,'ZData',z-10);
    view(2);
    grid on;
    hold on;
    for j=1:length(args.alpha)
        plot(args.Mu(1,j),args.Mu(2,j),'g^');
    end
    plot(debug.x(1,1:args.t),debug.x(2,1:args.t),'r');
    plot(debug.x(1,1),debug.x(2,1),'rs');
    plot(debug.trees(:,1),debug.trees(:,2),'sc');
    xlim([0 1])
    ylim([0 1])
end

