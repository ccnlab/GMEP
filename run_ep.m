function model=run_ep(model)
%
% A general purpose parallel EP algorithm that does parallel updates at
% separate cycles for the likelihood terms related to the observed data and
% possible non-conjugate prior terms. The following general model structure 
% is assumed:
%
% p(w,u,v) = prod_i p(y_i|Uw(:,i)'*w,Uv(:,i)'*v)   (likelihood terms)
%           *prod_j p(Uw(:,j)*w|Uu(:,j)'*u)         (prior terms)
%           *p(u)*p(v)                             (fixed priors)
%           *Z^{1}                                 (marginal likelihood)
% 
% The EP algorithm forms a posterior approximation q(w,u,v) by replacing 
% non-analytical likelihood or prior terms with analytical local
% approximations.
%
% Certain types of models may not require either likelihood approximations or
% prior approximations, and this is indicated by the fields runEP_lik and 
% runEP_pri in the input model structure. Scheduling the learning
% between likelihood and prior can be controlled by adjusting separate
% damping factors df_lik and df_pri as described below.
% 
% The input model structure requires the following model-specific functions:
%
% 1) qc=model.fh_cavity_moments(model,eta,cavity_type)
% This function computes the cavity distributions, whose form depend on 
% the chosen model specification.
%
% 2) [model,pos_def]=model.fh_model_update(model,site,qc,qt); 
% This function recomputes the posterior approximation using new site 
% parameters site, the cavity qc, and the tilted distribution qt. 
% This step requires model specific implementation.
% 
% The algorithm uses the following general purpose functions:
%
% 1) qt=tilted_moments(qc,y,eta,model_term_type,quad_opt)
% This function computes the tilted distribution using the cavity qc and
% optional data vector y (not usually required for prior terms).
% Model_term_type can be e.g. 'Gaussian', 'Laplace', or 'Probit'. 
% Parameter eta can be used to implement fractional updates, eta=1 results
% in regular EP.
%
% 2) site_new=site_update(qc,qt,eta,df);
% Computes the new site parameters resulting from the cavity qc and tilted
% distribution qt with a damping level (step-size) df.
% 
% Options in model.ep_opt:
% 
% ep_opt.niter   = max number of iterations
% ep_opt.tol     = site parameter tolerance level at convergence
% ep_opt.df_lik  = damping level for likelihood term updates
% ep_opt.df_pri  =  damping level for prior term updates 
%                  Set df_lik(1)/df_pri(1) for main the parameters 
%                  and df_lik(2)/df_pri(2) for optional hyperparameters.
% ep_opt.eta_lik = fraction parameter (0 1] for the likelihood updates
% ep_opt.eta_pri = fraction parameter (0 1] for the prior updates
%
% model.quad_opt = options for quadrature integrations 
%
%
% Pasi Jylänki 2014

ep_opt=model.ep_opt;
niter=ep_opt.niter;       % number of iterations
tol=ep_opt.tol;           % site parameter tolerance level at convergence
quad_opt=model.quad_opt;  % options for quadrature integrations

if ~model.runEP_lik && ~model.runEP_pri
  fprintf('All site terms Gaussian -> no EP required\n')
  return
end

if model.runEP_lik
  % damping factor for the likelihood terms
  df_lik=ep_opt.df_lik;
  
  % compute initial cavity distributions for the likelihood sites
  % (model-specific implementation)
  qc_lik=model.fh_cavity_moments(model,ep_opt.eta_lik,'lik');
  pos_def=all(qc_lik.pos_def);
  if ~pos_def
    fprintf('Ill-conditioned initial likelihood cavity distribution\n')
  end
else
  df_lik=0;
  fprintf('Likelihood terms Gaussian -> no EP required\n')
end

if model.runEP_pri
  % damping factor for the likelihood terms
  df_pri=ep_opt.df_pri;
  
  % compute initial cavity distributions for the likelihood sites
  % (model-specific implementation)
  qc_pri=model.fh_cavity_moments(model,ep_opt.eta_pri,'pri');
  pos_def=all(qc_pri.pos_def);
  if ~pos_def
    fprintf('Ill-conditioned initial prior cavity distribution\n')
  end
else
  df_pri=0;
  fprintf('Prior terms Gaussian -> no EP required\n')
end

if all(df_lik==0) && all(df_pri==0)
  fprintf('Unable to continue because step sizes set to zero\n')
  return
end

ds_lik=0; % monitor change in likelihood site parameters for convergence
ds_pri=0; % monitor change in prior site parameters for convergence

for iter=1:niter
  %%%%%%%%%%%%%%%%%%%%%%%%
  % parallel EP iterations
  
  if model.runEP_lik && any(df_lik>0)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EP update on the likelihood sites
    site_type='lik';
    eta=model.ep_opt.eta_lik;
    
    % compute the tilted moments (general implementation)
    qt_lik=tilted_moments(qc_lik,model.lik.y,eta,model.lik.type,quad_opt);
    
    pos_def=0;
    while ~pos_def && any(df_lik>1e-4)
      % try a parallel EP update by gradually decresing the step size until 
      % the new approximation and its cavity distributions are proper
      
      % compute site updates (general implementation)
      site_new=site_update(qc_lik,qt_lik,eta,df_lik);
      
      % update model structure and approximations (model-specific implementation)
      [model_new,pos_def]=model.fh_model_update(model,site_new,qc_lik,qt_lik);
      
      if ~pos_def
        % the new posterior approximation ill-conditioned -> increase damping
        df_lik=df_lik*0.9;
        fprintf('Ill-conditioned posterior approximation, decresing step size\n')
        continue
      end
      
      % compute new cavity distributions (model-specific implementation)
      qc_lik_new=model.fh_cavity_moments(model_new,eta,site_type);
      pos_def=all(qc_lik_new.pos_def);
      
      if ~pos_def
        % the new cavity distribution(s) ill-conditioned -> increase damping
        df_lik=df_lik*0.9;
        fprintf('Ill-conditioned cavity distribution, decresing step size\n')
        continue
      else
        % accept the new approximation
        model=model_new;
        qc_lik=qc_lik_new;
        ds_lik=site_new.ds; % site parameter changes [dnu1, dtau1, dnu2, dtau2]
      end
    end
  end
  
  if model.runEP_pri && any(df_pri>0)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EP update on the prior sites
    site_type='pri';
    eta=model.ep_opt.eta_pri;
    
    % compute the tilted moments (general implementation)
    qt_pri=tilted_moments(qc_pri,model.pri.y,eta,model.pri.type,quad_opt);
    
    pos_def=0;
    while ~pos_def && any(df_pri>1e-4)
      % try a parallel EP update by gradually decresing the step size until 
      % the new approximation and its cavity distributions are proper
      
      % compute site updates (general implementation)
      site_new=site_update(qc_pri,qt_pri,eta,df_pri);
      
      % update model structure and approximations (model-specific implementation)
      [model_new,pos_def]=model.fh_model_update(model,site_new,qc_pri,qt_pri);
      
      if ~pos_def
        % the new posterior approximation ill-conditioned -> increase damping
        df_pri=df_pri*0.9;
        fprintf('Ill-conditioned posterior approximation, decresing step size\n')
        continue
      end
      
      % compute new cavity distributions (model-specific implementation)
      qc_pri_new=model.fh_cavity_moments(model_new,eta,site_type);
      pos_def=all(qc_pri_new.pos_def);
      
      if ~pos_def
        % the new cavity distribution(s) ill-conditioned -> increase damping
        df_pri=df_pri*0.9;
        fprintf('Ill-conditioned cavity distribution, decresing step size\n')
        continue
      else
        % accept the new approximation
        model=model_new;
        qc_pri=qc_pri_new;
        ds_pri=site_new.ds; % site parameter changes [dnu1, dtau1, dnu2, dtau2]
      end
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % the marginal likelihood estimate
  logZ=model.logZq +model.lik.logZi +model.lik.logZp...
    +model.pri.logZi +model.pri.logZp;
  model.logZ=logZ;
  
  % approximate mean leave-one-out log predictive densities (mlpd) 
  % are also useful in monitoring convergence
  if isfield(model.lik,'logZt')
    mlpd=mean(model.lik.logZt);
    model.mlpd=mlpd;
  else
    mlpd=0;
  end
  fprintf('Iter %d, mlpd=%.3f, logZ=%.3f, ds_lik=[%s\b], ds_pri=[%s\b], df_lik=[%s\b], df_pri=[%s\b]\n',...
    iter,mlpd,logZ,sprintf('%.2f,',ds_lik),sprintf('%.2f,',ds_pri),...
    sprintf('%.2f,',df_lik),sprintf('%.2f,',df_pri))
  
  if all(ds_lik<tol) && all(ds_pri<tol)
    fprintf('Convergence within tolerance set by ''ep_opt.tol''\n')
    break;
  end
end
