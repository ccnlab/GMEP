% Main script for running GMEP
% 
% INPUT: 
% data: set of nS time series of ntime length ntime
% p_order: order of the MAR model
% structure_prior: specify the stuctured prior 
% n: number training time points
% nt: number testing time points
% hyper: structure containing initial points for likelihood, prior and hyperprior distributions
%
% OUTPUT:
% coeff_w: estimated coefficcients
% logZ: model evidence
% mlpd_ep: mean leave-one-out predictive density
% mlpd_train: mean log predictive density computed on the training data
% mlpd_test: mean log predictive density computed on the testing data
% causal_conf: causal configuration matrix, from row to column
%

clear all
close all
clc

normalize=1; % to normalize data zero mean and unit std 
to_binary=1; % to compute the binary causal configuration matrix

% Define the structured prior: 0->uniform Gaussian prior (default), 1->ARD
% prior, 2->lag-idependent prior.
structured_prior=2; 
         
%%%%%%%%%%%     
% INPUT TIME SERIES
% Load data, the variable data contains the set of time series 
% it is supposed to have this shape: [nTime, nSignal]
% filename=...        
% data=load(filename);

[ntime,nS]=size(data);
p_order=10; % order of the MAR model
p1=p_order+1;
M=ntime-p_order;
np=nS*p_order;

% normalization
data_std=1;
data_mean=0;
if normalize 
    data_std=std(data(:));
    data_mean=mean(data(:));
end
data=(data-data_mean)./data_std;
data=(data-data_mean)./data_std;

Y=reshape(data(p1:ntime,:),M,nS); % concatenate trials for unlagged observations
X=zeros(nS,p_order,M);
for k=1:p_order
    X(:,k,:)=reshape(data(p1-k:ntime-k,:)',nS,M); % concatenate for k-lagged observations
end
X=reshape(X,np,M)'; % design matrix

n1=1; % initial time point
n=500;  % number of training data points
nt=300; % number of testing data points
nin=size(X,2);  % number of inputs
nout=size(Y,2); % number of outputs

hyper.lik_inf_hyper=1;   % set 1 to integrate over hyperparameters
hyper.lik_mv0=0.5^2;     % prior mean likelihood
hyper.lik_Vv0=1.5^2;     % prior variance likelihood
hyper.pri_inf_hyper=1;   % set 1 to integrate over hyperparameters
hyper.pri_mw0=0.0;       % prior mean structured prior 
hyper.pri_Vw0=1.^2;      % prior variance structured prior
hyper.pri_mu0=-9.5;      % hyperprior mean 
hyper.pri_Vu0=3.5^2;     % hyperprior variance

x=squeeze(X(n1:n1+n-1,:));       % training design matrix
xt=squeeze(X(n1+n:n1+n+nt-1,:)); % testing design matrix

y=squeeze(Y(n1:n1+n-1,:));       % training output data 
yt=squeeze(Y(n1+n:n1+n+nt-1,:)); % testing output data

% Set up a linear model
clear model
model.type='linear';  % model type
d=nin+1;              % number of inputs plus a constant term
nw=d*nout;            % number of coefficients w

% set up the likelihood
model.lik.type='Gaussian'; % you can try Gaussian observation model
%model.lik.type='Laplace'; % or Laplace, which should be more heavy-tailed
model.lik.inf_hyper=hyper.lik_inf_hyper; 

if model.lik.inf_hyper
    % Set fixed Gaussian priors p(v)=N(mv0,Vv0) for the noise scale
    % hyperparamters v=log(h0), where h0 = Var(y|w) e.g. when p(y|w)=N(y|X*w,h0)
    % The log transformation could be generilized to any invertible
    % transformation
    model.lik.mv0=repmat(log(hyper.lik_mv0),nout,1);  % prior means
    model.lik.Vv0=repmat(hyper.lik_Vv0,nout,1);       % prior variances
else
    % fix the hyperparamters h=exp(v)
    model.lik.h0=(repmat(0.2,nout,1)).^2; % fixed noise variances (one for each output)
end

%%%%%%%%%%%%%%%%%%
% set up the prior
model.pri.type='Gaussian'; % you can try Gaussian priors
%model.pri.type='Laplace'; % or Laplace priors
model.pri.inf_hyper=hyper.pri_inf_hyper;  

% prior for the coefficients w (these remain fixed if inf_hyper=false)
model.pri.mw0=repmat(hyper.pri_mw0,d,nout);    % initial prior means
model.pri.Vw0=repmat(hyper.pri_Vw0/d,d,nout);  % initial prior variances 

% set a minimum level for site precision parameters (optional)
% this improves stability of the algorithm, by keeping prior term
% precision parameters positive
model.pri.tau_w_min=1/10^2;

if model.pri.inf_hyper
    % Assign coefs to different possibly overlapping groups and
    % assign one variance hyperparameter u for each group.
    % Couple the different outputs using the same hypers u.
    %
    % Note that without coupling through likelihood or prior hyperparameters
    % the linear models associated with the different outputs could be simply
    % inferred separately
    switch structured_prior
        case 1,
            ng=nout*p_order; % ARD prior
        case 2,
            ng=nout;         % lag-independent prior
        otherwise,
            ng=1;            % uniform Gaussian prior (default)
    end

    nu=(ng+1)*nout; % separate group for the bias -> +1
    Uu=zeros(nu,d*nout);
    Ui=zeros(ng,d);
    % form structure for one output
    for i1=1:ng
     Ui(i1,i1:ng:nin)=1;
    end
    % repeat the same structure for all outputs
    for i1=1:nout
     Uu( ng*(i1-1) +(1:ng), d*(i1-1) +(1:d) ) = Ui;
     % and bias separately for each output
     Uu(nout*ng+i1,(i1-1)*d+d) = 1;
    end
    model.pri.Uu=Uu;

    % Each coefficients needs to be assigned at least to one group to keep
    % the algorithm stable.

    % Set fixed Gaussian priors p(u)=N(mu0,Vu0) for the prior scale
    % hyperparamters u=log(h0), where h0 = Var(w) e.g. when p(w)=N(w|0,h0)
    % The log transformation could be generilized to any invertible
    % transformation
    model.pri.mu0=repmat(hyper.pri_mu0,nu,1);
    model.pri.Vu0=repmat(hyper.pri_Vu0,nu,1);
else
    % fix the hyperparamters h0=exp(v)
    model.pri.h0=repmat(0.1,1,nout).^2; % fixed prior variances (one for each output)
end

% initialize the model
model = approximation_hglm(model,'init',x,y);

% parameters for EP algorithm
model.ep_opt.tol=1e-3;          % tolerance level for convergence
model.ep_opt.eta_lik=1.0;       % fraction parameter for likelihood terms
model.ep_opt.eta_pri=0.9;       % fraction parameter for prior terms
model.ep_opt.df_lik=[0.7 0.5];  % damping factor 0<=df<=1 for likelihood terms
                                % df_lik(1) for w and df_lik(2) hypers v
model.ep_opt.df_pri=[0.7 0.5];  % non-zero step-size for the prior terms
                                % df_pri(1) for w and df_pri(2) hypers u

% run ep for the likelihood terms first to get a good data fit
model.ep_opt.df_pri=0.0;        % suppress updates for prior terms
if model.lik.inf_hyper
    % If likelihood hyperparamters present, update first few times only 
    % the coefficient
    model.ep_opt.niter=5;
    model.ep_opt.df_lik=[0.7 0.0];
    model=run_ep(model);
end
model.ep_opt.niter=50;
model.ep_opt.df_lik=[0.8 0.5];
model=run_ep(model);

% run ep also for the prior terms
% Update first few times only the prior terms
model.ep_opt.niter=10;
model.ep_opt.df_lik=[0.0 0.0];
model.ep_opt.df_pri=[0.0 0.7];
model=run_ep(model);

model.ep_opt.df_pri=[0.5 0.2];
model=run_ep(model);

% Update all terms simultaneously
model.ep_opt.niter=50;
model.ep_opt.df_lik=[0.8 0.6];
model.ep_opt.df_pri=[0.8 0.6];
model=run_ep(model);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qp=predict(model,x,y);
% testing data
qpt=predict(model,xt,yt);
                
% Save some measures
coeff_w=model.qw.m(1:nin,:); % Estimated coefficcients
logZ=model.logZ; % Model evidence
mlpd_ep=model.mlpd; % Mean leave-one-out predictive density
mlpd_train=mean(qp.logpy(:)); % Mean log predictive density computed on the training data
mlpd_test=mean(qpt.logpy(:)); % Mean log predictive density computed on the testing data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate of the causal configuration matrix
if to_binary && structured_prior==2
    
    su_tmp=sqrt(diag(model.qu.S));
    mu=zeros(nS,nS);
    su=zeros(nS,nS);                       
    for i_w=1:nS^2
        mu(i_w)=model.qu.m(i_w);
        su(i_w)=su_tmp(i_w);
    end                    
    % Normalize mu and su
    mu_norm=(mu-min(mu(:)))/(max(mu(:))-min(mu(:)));
    su_norm=(su-min(su(:)))/(max(su(:))-min(su(:)));
    % Comparison between mean ans std
    causal_conf=su_norm<mu_norm;
    causal_conf(diag(ones(nS,1))==1)=0;
    
    disp('Estimated causal configuration matrix (direction:row->column):')
    disp(causal_conf)
end                                
