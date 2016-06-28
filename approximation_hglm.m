function model = approximation_hglm(model,task,X,Y)
%
% Model specific EP computations for a hierarchical generalized linear model
% 
% p(y,w,v,u) = prod_i p(y(i)|Uw(:,i)*w,Uv(:,i)'*v) 
%            * prod_j p(w(j)|Uu(:,j)'*u)*p(v)*p(u)
% 
% where transformations Uw are constructed using X, and Uv and Uu can be
% specified by the user. v and u are optional hyperparamters related to the
% likelihood and prior terms respectively, and they are given Gaussian
% priors p(v) and p(u).
%
% Indepedent Gaussian approximations are constructed for the model parameters
% q(w)=N(model.qw.m,model.qw.S)
% q(v)=N(model.qv.m,model.qv.S)
% q(u)=N(model.qu.m,model.qu.S)
%
% In the EP algorithm the likelihood terms p(y(i)|Uw(:,i) are updated in a
% separater parallel sweep from the prior terms p(w(j)|Uu(:,j).
% 
% The following subfunctions are required for subsequent use of the 
% attached run_ep.m algorithm:
%
% 1) init_model initializes the model parameters and the approximations
%
% 2) compute_q (re)computes the approximations q(w), q(v), and q(u)
%
% 3) cavity_moments computes the cavity distributions of the transformed 
% random variables z1=Uw(:,i)'*w, and z2=Uv(:,i)'*v or z2=Uu(:,j)'*u
%
% 4) model_update replaces the old model paramters with the new ones and
% recomputes q(w) and q(v) or (q(w) and q(u)) after each parallel EP update
% 
% 5) qf_pred computes the latents z1=X(:,i)'*w, and z2=Uv(:,i)'*v that are
% required in the predictions after convergence of the EP algorithm.
%
% Pasi Jylänki 2014

switch task
  case 'init'
    
    model=init_model(model,X,Y);
    model=compute_q(model,'all');
    
    model.fh_init_model = @init_model;
    model.fh_compute_q = @compute_q;
    model.fh_cavity_moments = @cavity_moments;
    model.fh_model_update = @model_update;
    model.fh_qf_pred = @qf_pred;
    model.fh_reset_model = @reset_model;
  otherwise
    fprintf('Unknown task\n')
end

end

function model=init_model(model,X,Y)
%%%%%%%%%%%%%%%%%%%%%%
% initialize the model

[n,nout]=size(Y);
nin=size(X,2);
d=nin+1;

model.n=n;
model.nout=nout;
model.nin=nin;
model.d=d;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% init likelihood approximation
nlik=nout*n;
model.nlik=nlik;

% set up likelihood approximation for coefficients w
if ~isfield(model.lik,'Uw')
  % transformation matrix for coefficients w
  model.lik.Uw=[X ones(n,1)]'; % append with constant coefficient
  model.lik.iUw=repmat(1:n,1,nout);
end

% marginal likelihood contribution
model.lik.logZi=0; % initial site normalizations

% initialize site parameters to zero
model.lik.nu_w=zeros(1,nlik);
model.lik.tau_w=zeros(1,nlik);

switch model.lik.type
  % likelihood specific settings here
  case 'Gaussian'
    if model.lik.inf_hyper
      model.runEP_lik=true;
    else
      % no EP required
      model.runEP_lik=false;
      % No EP required with Gaussian likelihood with known hypers
      
      % Fixed noise variance parameters
      Vs2=model.lik.h0(:)';
      model.lik.vi=Vs2;
      
      % fixe site paramters to the known values
      nu_w=bsxfun(@rdivide,Y,Vs2);
      model.lik.nu_w=nu_w(:)';
      tau_w=repmat(1./Vs2,n,1);
      model.lik.tau_w=tau_w(:)';
      
      %           logZi = bsxfun(@plus, -0.5*log(2*pi) -0.5*log(s2),...
      %             -0.5*bsxfun(@rdivide,Y.^2,s2));
      %           logZi = sum(sum(logZi));
      
      % marginal likelihood contribution
      model.lik.logZi = -nlik/2*log(2*pi) -0.5*n*sum(log(Vs2)) -0.5*sum(Y.^2 * (1./Vs2'));
    end
  case 'Laplace'
    model.runEP_lik=true;
    if ~model.lik.inf_hyper
      % Fixed noise variance parameters
      vi=model.lik.h0(:)';
      model.lik.vi=vi;
      v=repmat(vi,n,1);
      model.lik.v=v(:);
    end
  case 'Probit'
    model.runEP_lik=true;
    model.lik.inf_hyper=0; % no likelihood hypers
    Y=Y*2-1; % transform binary targets to [-1 1]
    model.lik.vi=[];
    model.lik.v=[];
  otherwise
    fprintf('Unknown likelihood type\n')
    return
end
model.lik.y=Y(:);

if model.lik.inf_hyper
  % infer likelihood hyperparamters v
  if ~isfield(model.lik,'Uv')
    % Define transformation Uv_i for the likelihood hypers,
    % where output j uses Uv_i(:,j).
    % By default each output uses its own hyperparameter:
    Uv_i=eye(nout);
    model.lik.Uv_i=Uv_i;
    
    dv=size(Uv_i,1);
    model.lik.dv=dv;
    
    % for ep: expand Uv_pred across likelihood terms
    model.lik.Uv=zeros(dv,nlik);
    for i1=1:nout
      model.lik.Uv(:,(i1-1)*n+1:(i1-1)*n+n)=repmat(Uv_i(:,i1),1,n);
    end
  else
    % give Uv as a n*dv*nout matrix
    dv=size(model.lik.Uv,2);
    model.lik.Uv=reshape(permute(model.lik.Uv,[2,1,3]),dv,n*nout);
    model.lik.dv=dv;
  end
  
  % fixed priors for likelihood hyperparameters v
  if ~isfield(model.lik,'mv0')
    model.lik.mv0=repmat(log(0.1^2),nout,1);  % prior means
    model.lik.Vv0=repmat(1.0^2,nout,1);       % prior variances
  end
  mv0=model.lik.mv0(:);
  Vv0=model.lik.Vv0(:);
  
  nu_v0=mv0./Vv0;
  tau_v0=1./Vv0;
  model.lik.nu_v0=nu_v0;
  model.lik.tau_v0=tau_v0;
  
  % site parameters for noise hyperparameters  v
  model.lik.nu_v=zeros(1,nlik);
  model.lik.tau_v=zeros(1,nlik);
  
  % marginal likelihood contribution from fixed hyper priors
  model.lik.logZp=-dv/2*log(2*pi) -0.5*sum(log(Vv0(:))) -0.5*sum(mv0(:).^2 ./Vv0(:));
else
  % marginal likelihood contribution when no prior terms associated with
  % likelihood hyperparameters
  model.lik.logZp=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% init prior approximation

% set up prior approximation for coefficients w
% initial prior means and variance need to be supplied by the user
mw0=model.pri.mw0(:);
Vw0=model.pri.Vw0(:);

% initial site parameters
nu_w=mw0./Vw0;
model.pri.nu_w=nu_w';
tau_w=1./Vw0;
model.pri.tau_w=tau_w';

if ~isfield(model.pri,'Uw')
  % model.pri.Uw=eye(nw,nsites); % each column of Uw corresponds to one prior term  
  model.pri.Uw='I'; % use short hand for identiy matrix
  npri=d*nout;
end

model.npri=npri;
model.pri.y=[];

% marginal likelihood contribution
model.pri.logZi=0; % initial site normalizations

switch model.pri.type
  % model specific settings here
  case 'Gaussian'
    if model.pri.inf_hyper
      model.runEP_pri=true;
    else
      % No EP required with a Gaussian prior
      model.runEP_pri=false;
      % marginal likelihood contribution
      model.pri.logZi=-npri/2*log(2*pi) -0.5*sum(log(Vw0(:))) -0.5*sum(mw0(:).^2 ./Vw0(:));
    end
  case 'Laplace'
    model.runEP_pri=true;
    if ~model.pri.inf_hyper
      % Fixed prior variance parameters
      vi=model.pri.h0(:)';
      model.pri.vi=vi;
      v=repmat(vi,d,1);
      model.pri.v=v(:);
    end
end

if model.pri.inf_hyper
  % Infer hyper paramters
  
  % make sure the prior transformation is defined
  if ~isfield(model.pri,'Uu')
    if isfield(model.pri,'group')
      % user defined groups
      
      % create transformation matrix for one output
      ng=numel(model.pri.group);
      Ui=zeros(ng,d);
      for i1=1:ng
        Ui(i1,model.pri.group{i1})=1;
      end
      
      % use the same configuration for all output components
      model.pri.Uu=repmat(Ui,1,nout);
    else
      % use one hyperparameter by default
      model.pri.Uu=ones(1,npri);
      model.pri.mu0=log(0.1^2);
      model.pri.Vu0=1^2;
    end
  end
  du=size(model.pri.Uu,1);
  model.pri.du=du;
  
  % fixed hyperpriors for noise hyperparameters  v
  if ~isfield(model.pri,'mu0')
    model.pri.mu0=repmat(log(0.1^2),du,1);  % prior means
    model.pri.Vu0=repmat(1.0^2,du,1);       % prior variances
  end
  mu0=model.pri.mu0(:);
  Vu0=model.pri.Vu0(:);
  
  nu_u=mu0./Vu0;
  tau_u=1./Vu0;
  model.pri.nu_u0=nu_u;
  model.pri.tau_u0=tau_u;
  
  % site parameters for noise hyperparameters v
  model.pri.nu_u=zeros(1,npri);
  model.pri.tau_u=zeros(1,npri);
  
  % marginal likelihood contribution from fixed hyper priors
  model.pri.logZp=-du/2*log(2*pi) -0.5*sum(log(Vu0(:))) -0.5*sum(mu0(:).^2 ./Vu0(:));
else
  % marginal likelihood contribution when no prior terms associated with
  % likelihood hyperparameters
  model.pri.logZp=0;
end

%%%%%%%%%%%%%%%%%%%%
% quadrature options
np=201;
model.quad_opt.n_quad_points=np;

% integration limits wrt cavity std
qlim=[-10,10];
model.quad_opt.quad_lim=qlim;

% normalized grid for Gaussian integrals
qval=qlim(1)+(0:np-1)*(qlim(2)-qlim(1))/(np-1);
model.quad_opt.quad_grid=qval;

% for simplicity, use Simpson's rule with a fixed grid
%du=uv(2)-uv(1);
%qw=[0.5; ones(np-2,1); 0.5]*du; % trapezoid
%qw=[1 repmat([4 2],1,(np-3)/2) 4 1]'*(du/3); % Simpson
tmp=1/3;
qw=ones(np,1);
qw([1 np])=tmp;
qw(2:2:np-1)=4*tmp;
qw(3:2:np-2)=2*tmp;
model.quad_opt.quad_weights=qw;
end
% end of init_model
%%%%%%%%%%%%%%%%%%%

function [model,pos_def]=compute_q(model,q_type)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Gaussian approximations
% q(w) for model coefficients f=X*w
% q(v) for optional likelihood hyperparamters
% q(u) for optional prior hyperparamters

d=model.d;
n=model.n;
nout=model.nout;
pos_def=1;

if strcmp(q_type,'all') || strcmp(q_type,'lik') || strcmp(q_type,'pri')
  %%%%%%%%%%%%%%%%
  % recompute q(w)
  nu_p=model.pri.nu_w;
  tau_p=model.pri.tau_w;
  Up=model.pri.Uw;
  
  nu_l=model.lik.nu_w;
  tau_l=model.lik.tau_w;
  Ul=model.lik.Uw;
  
  % full covariance approximation not implemented
  %   if model.lik.full
  %     dv=model.lik.dv;
  %     tau_wv=model.lik.tau_wv;
  %   else
  
  % factorized approximation for the likelihood hypers q(w,v)=q(w)q(v)
  h=zeros(d,nout);
  Q=zeros(d,d,nout);
  m=zeros(d,nout);
  S=zeros(d,d,nout);
  logZq=nout*0.5*d*log(2*pi);
  
  for i1=1:nout
    % prior contribution
    i0=(i1-1)*d;
    ii1=i0+1:i0+d;
    if strcmp(Up,'I')
      % one prior term for each coef
      h(:,i1)=nu_p(ii1)';
      Q(:,:,i1)=diag(tau_p(ii1));
    else
      % tau_w should be positive
      % Uw is also assumed block diagonal
      % in the factorized approximation
      % Uw0=randn(d,model.npri);
      
      fprintf('General prior Uw not implemented yet\n')
      return
      
      %Lw=bsxfun(@times,Uw0(iw,:,i1),sqrt(tau_w0));
      %Qw(:,:,i1)=Lw*Lw';
    end
    
    % likelihood contribution
    i0=(i1-1)*n;
    ii1=i0+1:i0+n;
    h(:,i1)=h(:,i1)+Ul*nu_l(ii1)';
    tau_i=tau_l(ii1);
    ii=tau_i>=0;
    L=bsxfun(@times,Ul(:,ii),sqrt(tau_i(ii)));
    Q(:,:,i1)=Q(:,:,i1)+L*L';
    ii=~ii;
    if any(ii)
      L=bsxfun(@times,Ul(:,ii),sqrt(abs(tau_i(ii))));
      Q(:,:,i1)=Q(:,:,i1)-L*L';
    end
    
    %     if strcmp(Uwl,'X')
    %       % one prior term for each coef
    %       Lw=bsxfun(@times,X',sqrt(tau_wl(ii)));
    %       Qw(:,:,i1)=Qw(:,:,i1)+Lw*Lw';
    %     else
    %       % tau_w should be positive
    %       % Uw is also assumed block diagonal
    %       % in the factorized approximation
    %
    %       sprintf('General likelihood Uw not implemented yet\n')
    %       break
    %
    %       %Lw=bsxfun(@times,Uw(ii,ii,i1),sqrt(tau_w(ii)));
    %       %Tw(:,:,i1)=Lw*Lw';
    %     end
    
    [L,p]=chol(Q(:,:,i1),'lower');
    if p~=0
      fprintf('Ill-conditioned posterior covariance\n')
      pos_def=0;
      return
    else
      A=L\eye(d);
      S(:,:,i1)=A'*A;
      m(:,i1)=L'\(L\(h(:,i1)));
      logZq=logZq +0.5*(m(:,i1)'*h(:,i1)) -sum(log(diag(L)));
    end
  end
  model.qw.h=h;
  model.qw.Q=Q;
  model.qw.m=m;
  model.qw.S=S;
  model.qw.logZq=logZq;
end

if model.lik.inf_hyper && (strcmp(q_type,'all') || strcmp(q_type,'lik'))
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % recompute q(v) related to likelihood sites
  dd=model.lik.dv;
  
  nu_p=model.lik.nu_v0;
  tau_p=model.lik.tau_v0;
  nu_l=model.lik.nu_v;
  tau_l=model.lik.tau_v;
  U=model.lik.Uv;
  
  h=nu_p +U*nu_l';
  Q=diag(tau_p);
  ii=tau_l>=0;
  L=bsxfun(@times,U(:,ii),sqrt(tau_l(ii)));
  Q=Q+L*L';
  ii=~ii;
  if any(ii)
    L=bsxfun(@times,U(:,ii),sqrt(abs(tau_l(ii))));
    Q=Q-L*L';
  end
  
  [L,p]=chol(Q,'lower');
  if p~=0
    sprintf('Ill-conditioned posterior covariance\n')
    pos_def=0;
    return
  else
    m=L'\(L\h);
    A=L\eye(dd);
    S=A'*A;
    logZq=0.5*dd*log(2*pi) +0.5*(m'*h) -sum(log(diag(L)));
  end
  
  model.qv.h=h;
  model.qv.Q=Q;
  model.qv.m=m;
  model.qv.S=S;
  model.qv.logZq=logZq;
end

if model.pri.inf_hyper && (strcmp(q_type,'all') || strcmp(q_type,'pri'))
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % recompute q(u) related to prior sites
  dd=model.pri.du;
  
  nu_p=model.pri.nu_u0;
  tau_p=model.pri.tau_u0;
  nu_l=model.pri.nu_u;
  tau_l=model.pri.tau_u;
  U=model.pri.Uu;
  
  h=nu_p +U*nu_l';
  Q=diag(tau_p);
  ii=tau_l>=0;
  L=bsxfun(@times,U(:,ii),sqrt(tau_l(ii)));
  Q=Q+L*L';
  ii=~ii;
  if any(ii)
    L=bsxfun(@times,U(:,ii),sqrt(abs(tau_l(ii))));
    Q=Q-L*L';
  end
  
  [L,p]=chol(Q,'lower');
  if p~=0
    sprintf('Ill-conditioned posterior covariance\n')
    pos_def=0;
    return
  else
    m=L'\(L\h);
    A=L\eye(dd);
    S=A'*A;
    logZq=0.5*dd*log(2*pi) +0.5*(m'*h) -sum(log(diag(L)));
  end
  
  model.qu.h=h;
  model.qu.Q=Q;
  model.qu.m=m;
  model.qu.S=S;
  model.qu.logZq=logZq;
end

% marginal likelihood contribution
logZq=model.qw.logZq;
if model.lik.inf_hyper
  logZq=logZq+model.qv.logZq;
end
if model.pri.inf_hyper
  logZq=logZq+model.qu.logZq;
end
model.logZq=logZq;
end
% end of compute_q
%%%%%%%%%%%%%%%%%%

function qc=cavity_moments(model,eta,cavity_type)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Gaussian cavity distributions
% q(z1) for latent variables z1=Uw'*w using q(w) 
% q(z2) for latent variables z2=Uu'*u or z2=Uv'*v related 
% to hyperparameters using q(v) or q(u) respectively.

qc.type=cavity_type;
switch cavity_type
  case 'lik'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % cavity for the likelihood sites
    nout=model.nout;
    n=model.n;
    
    % cavity for coefficients w
    U=model.lik.Uw;
    m=model.qw.m;
    S=model.qw.S;
    tau=model.lik.tau_w(:);
    nu=model.lik.nu_w(:);
    
    mz=zeros(n*nout,1);
    Vz=zeros(n*nout,1);
    for i1=1:nout
      ii1=(i1-1)*n+1:(i1-1)*n+n;
      mz(ii1)=U'*m(:,i1);
      Vz(ii1)=sum(U.*(S(:,:,i1)*U),1)';
    end
    
    Vc=1./(1./Vz-eta*tau);
    mc=Vc.*(mz./Vz-eta*nu);
    
    qc.mz1=mc;
    qc.Vz1=Vc;
    qc.nu1=nu;
    qc.tau1=tau;
    qc.logZr=0.5*(mc.^2 ./Vc +log(Vc) -mz.^2 ./Vz -log(Vz));
    if any(Vc<=0)
      qc.pos_def(1)=0;
    else
      qc.pos_def(1)=1;
    end
    
    if isfield(model.lik,'tau_w_min')
      qc.tau_min1=model.lik.tau_w_min;
    end
    
    if model.lik.inf_hyper
      % cavity for likelihood hypers v
      U=model.lik.Uv;
      m=model.qv.m;
      S=model.qv.S;
      tau=model.lik.tau_v(:);
      nu=model.lik.nu_v(:);
      
      mz=U'*m;
      Vz=sum(U.*(S*U),1)';
      
      Vc=1./(1./Vz-eta*tau);
      mc=Vc.*(mz./Vz-eta*nu);
      
      qc.mz2=mc;
      qc.Vz2=Vc;
      qc.nu2=nu;
      qc.tau2=tau;
      qc.logZr=qc.logZr +0.5*(mc.^2 ./Vc +log(Vc) -mz.^2 ./Vz -log(Vz));
      if any(Vc<=0)
        qc.pos_def(2)=0;
      else
        qc.pos_def(2)=1;
      end
      
      if isfield(model.lik,'tau_v_min')
        qc.tau_min2=model.lik.tau_v_min;
      end
    else
      qc.v=model.lik.v;
    end
    
  case 'pri'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % cavity for the prior sites
    
    % cavity for coefficients w
    U=model.pri.Uw;
    m=model.qw.m;
    S=model.qw.S;
    tau=model.pri.tau_w(:);
    nu=model.pri.nu_w(:);
    
    if strcmp(U,'I')
      % identity transformation
      
      nout=model.nout;
      d=size(m,1);
      mz=zeros(d*nout,1);
      Vz=zeros(d*nout,1);
      for i1=1:nout
        ii1=(i1-1)*d+1:(i1-1)*d+d;
        mz(ii1)=m(:,i1);
        Vz(ii1)=diag(S(:,:,i1));
      end
    else
      fprintf('General prior Uw not implemented yet\n')
      return
    end
    
    Vc=1./(1./Vz-eta*tau);
    mc=Vc.*(mz./Vz-eta*nu);
    
    qc.mz1=mc;
    qc.Vz1=Vc;
    qc.nu1=nu;
    qc.tau1=tau;
    qc.logZr=0.5*(mc.^2 ./Vc +log(Vc) -mz.^2 ./Vz -log(Vz));
    if any(Vc<=0)
      qc.pos_def(1)=0;
    else
      qc.pos_def(1)=1;
    end
    
    if isfield(model.pri,'tau_w_min')
      qc.tau_min1=model.pri.tau_w_min;
    end
    
    if model.pri.inf_hyper
      % cavity for prior hypers u
      U=model.pri.Uu;
      m=model.qu.m;
      S=model.qu.S;
      tau=model.pri.tau_u(:);
      nu=model.pri.nu_u(:);
      
      mz=U'*m;
      Vz=sum(U.*(S*U),1)';
      
      Vc=1./(1./Vz-eta*tau);
      mc=Vc.*(mz./Vz-eta*nu);
      
      qc.mz2=mc;
      qc.Vz2=Vc;
      qc.nu2=nu;
      qc.tau2=tau;
      qc.logZr=qc.logZr +0.5*(mc.^2 ./Vc +log(Vc) -mz.^2 ./Vz -log(Vz));
      if any(Vc<=0)
        qc.pos_def(2)=0;
      else
        qc.pos_def(2)=1;
      end
      
      if isfield(model.pri,'tau_u_min')
        qc.tau_min2=model.pri.tau_u_min;
      end
    else
      qc.v=model.pri.v;
    end
end

end
% end of cavity_moments
%%%%%%%%%%%%%%%%%%%%%%%

function [model,pos_def]=model_update(model,site,qc,qt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A wrapper that updates the site parameters to the model structure
% and recomputes the approximation using compute_q

switch site.type
  case 'lik'
    % insert new site paramters to model structure
    model.lik.nu_w=site.nu1';
    model.lik.tau_w=site.tau1';
    
    if isfield(site,'nu2')
      model.lik.nu_v=site.nu2';
      model.lik.tau_v=site.tau2';
    end
    
    % recompute approximations
    [model,pos_def]=compute_q(model,'lik');
    
    % marginal likelihood contribution
    model.lik.logZt=qt.logZ;
    model.lik.logZr=qc.logZr;
    model.lik.logZi=sum(qt.logZ)+sum(qc.logZr);
  case 'pri'
    % insert new site paramters to model structure
    model.pri.nu_w=site.nu1';
    model.pri.tau_w=site.tau1';
    
    if isfield(site,'nu2')
      model.pri.nu_u=site.nu2';
      model.pri.tau_u=site.tau2';
    end
    
    % recompute approximations
    [model,pos_def]=compute_q(model,'pri');
    
    % marginal likelihood contribution
    model.pri.logZt=qt.logZ;
    model.pri.logZr=qc.logZr;
    model.pri.logZi=sum(qt.logZ)+sum(qc.logZr);
end

end

function qp=qf_pred(model,x)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the predictive distribution of the latent function values f=X*w

nout=model.nout;
nt=size(x,1);

x=[x ones(nt,1)]; % append constant term
qp.mf=x*model.qw.m;
qp.Vf=zeros(nt,nout);
for i1=1:nout
  qp.Vf(:,i1)=sum((x*model.qw.S(:,:,i1)).*x,2);
end

if model.lik.inf_hyper
  % the latent values related to hyperparameters h=Uv'*v
  
  if isfield(model.lik,'Uvt')
    % user specified Uvt should be a nt*dv*nout matrix
    Uvt=model.lik.Uvt;
  else
    Uv_i=model.lik.Uv_i;
    dv=size(Uv_i,1);
    Uvt=zeros(nt,dv,nout);
    for i1=1:nout
      Uvt(:,:,i1)=repmat(Uv_i(:,i1)',nt,1);
    end
  end
  
  qp.mh=zeros(nt,nout);
  qp.Vh=zeros(nt,nout);
  for i1=1:nout
    qp.mh(:,i1)=Uvt(:,:,i1)*model.qv.m;
    qp.Vh(:,i1)=sum((Uvt(:,:,i1)*model.qv.S).*Uvt(:,:,i1),2);
  end
else
  % likelihood hyperparameters h are known
  qp.h=repmat(model.lik.vi,nt,1);
end
end

function model=reset_model(model,site_type)

if strcmp(site_type,'all') || strcmp(site_type,'lik')
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  % re-init likelihood prior
  if model.lik.inf_hyper
    % infer likelihood hyperparamters v
    dv=model.lik.dv;
    
    % set new fixed hyper prior parameters
    mv0=model.lik.mv0(:)';
    Vv0=model.lik.Vv0(:)';
    model.lik.nu_v0=mv0./Vv0;
    model.lik.tau_v0=1./Vv0;
    
    % marginal likelihood contribution from fixed hyper priors
    model.lik.logZp=-dv/2*log(2*pi) -0.5*sum(log(Vv0(:))) -0.5*sum(mv0(:).^2 ./Vv0(:));
  else
    switch model.lik.type
      % likelihood specific settings here
      case 'Gaussian'
        % Fixed noise variance parameters
        Vs2=model.lik.h0(:)';
        model.lik.vi=Vs2;
        
        % fix the site parameters to the known values
        n=model.n;
        Y=reshape(model.lik.y,n,model.nout);
        nu_w=bsxfun(@rdivide,Y,Vs2);
        model.lik.nu_w=nu_w(:)';
        tau_w=repmat(1./Vs2,n,1);
        model.lik.tau_w=tau_w(:)';
        
        % marginal likelihood contribution
        model.lik.logZi = -model.nlik/2*log(2*pi) -0.5*n*sum(log(Vs2))...
          -0.5*sum(Y.^2 * (1./Vs2'));
      case 'Laplace'
        % Fixed noise variance parameters
        vi=model.lik.h0(:)';
        model.lik.vi=vi;
        v=repmat(vi,model.n,1);
        model.lik.v=v(:);
    end
    
    % marginal likelihood contribution when no prior terms associated with
    % likelihood hyperparameters
    model.lik.logZp=0;
  end
end

if strcmp(site_type,'all') || strcmp(site_type,'pri')
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % re-init prior approximation
  if model.pri.inf_hyper
    % Infer prior hyper parameters
    du=model.pri.du;
    
    % set new fixed hyper prior parameters
    mu0=model.pri.mu0(:);
    Vu0=model.pri.Vu0(:);
    model.pri.nu_u0=mu0./Vu0;
    model.pri.tau_u0=1./Vu0;
    
    % marginal likelihood contribution from fixed hyper priors
    model.pri.logZp=-du/2*log(2*pi) -0.5*sum(log(Vu0(:))) -0.5*sum(mu0(:).^2 ./Vu0(:));
  else
    % fixed prior hyperparameters
    switch model.pri.type
      case 'Gaussian'
        % set new prior means and variances
        mw0=model.pri.mw0(:)';
        Vw0=model.pri.Vw0(:)';
        model.pri.nu_w=mw0./Vw0;
        model.pri.tau_w=1./Vw0;
        
        % marginal likelihood contribution
        model.pri.logZi=-npri/2*log(2*pi) -0.5*sum(log(Vw0(:))) -0.5*sum(mw0(:).^2 ./Vw0(:));
      case 'Laplace'
        % Fixed prior variance parameters
        vi=model.pri.h0(:)';
        model.pri.vi=vi;
        v=repmat(vi,model.d,1);
        model.pri.v=v(:);
    end
    
    % marginal likelihood contribution when no prior terms for the prior hyperparameters
    model.pri.logZp=0;
  end
end

% recompute posterior approximation
if strcmp(site_type,'lik')
  model=model.fh_compute_q(model,'lik');
elseif strcmp(site_type,'pri')
  model=model.fh_compute_q(model,'pri');
else
  model=model.fh_compute_q(model,'all');
end

end
