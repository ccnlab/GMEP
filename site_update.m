function site=site_update(qc,qt,eta,dfv)
%
% Computes site stabilized site updates using cavity distribution qc and
% tilted distribution qt.
%
% Updates are computed always for primary paramters 
% qc(z1) = N(qc.mz1,qc.Vz1)
% qt(z1) = N(qt.mz1,qt.Vz1)
%
% If qt and qc have fields related to secondary hyperparamters z1
% qc(z2) = N(qc.mz2,qc.Vz2)
% qt(z2) = N(qt.mz2,qt.Vz2)
%
% The scakar sites are Gaussians defined by their natural exponential parameters:
% t(z1) = Z1*exp(nu1*z1 -0.5*tau1*z1^2)
% site.nu1 = site location
% site.tau1 = site precision
%
% The site precision parameters tau1 and tau2 can be constrained to strictly
% positive values by defining qc.tau_min1 as a lower limit for them. In
% practice this is done by replacing the tilted variances qt.Vz by smaller 
% values so that tau >= qc.tau_min. The tilted means qt.mz1 are matched
% exactly. As a result the approximation may slightly underestimate the local
% variances.
%
% Different damping (step size) factors can be assigned (nu1,tau1) and (nu2,tau2)
% in dfv(1) and dfv(2), respectively.
% 
% Pasi Jylänki 2014

% site update for tranformed variable z1
df=dfv(1);
if isfield(qc,'tau_min1')
  tau_min=qc.tau_min1;
else
  tau_min=[];
end

nu=qc.nu1;
tau=qc.tau1;
tau_new=tau;
nu_new=nu;

Vzt=qt.Vz1;
tau_zt=1./Vzt;

% find site terms with proper tilted moments
ii1=isfinite(Vzt) & Vzt>0 & isfinite(qt.logZ);

% update the site location parameters
tau_new0=tau;
tau_new0(ii1)=(tau_zt(ii1)-1./qc.Vz1(ii1))/eta;
tau_new(ii1)=(1-df)*tau(ii1) +df*tau_new0(ii1); % damped update
%tau_new(ii1)=(1-df)*tau(ii1) +(df/eta)*(tau_zt(ii1) -1./qc.Vz1(ii1));

if ~isempty(tau_min)
  % constrain the tilted variances if the site precisions get too small: tau < tau_min
  ii2=tau_new<tau_min;
  if any(ii2)
    tau_zt(ii2) = 1./qc.Vz1(ii2) +(eta/df)*(tau_min+(df-1)*tau(ii2));
    tau_new0(ii2) = (tau_zt(ii2)-1./qc.Vz1(ii2))/eta;
    tau_new(ii2)=(1-df)*tau(ii2) +df*tau_new0(ii2);
    %tau_new(ii2)=(1-df)*tau(ii2) +(df/eta)*(tau_zt(ii2) -1./qc.Vz1(ii2));
  end
end

% update the site location parameters
nu_new0=nu;
nu_new0(ii1)=(qt.mz1(ii1).*tau_zt(ii1) -qc.mz1(ii1)./qc.Vz1(ii1))/eta;
nu_new(ii1)=(1-df)*nu(ii1) +df*nu_new0(ii1);
%nu_new(ii1)=(1-df)*nu(ii1) +(df/eta)*(qt.mz1(ii1).*tau_zt(ii1) -qc.mz1(ii1)./qc.Vz1(ii1));

ds=[max(abs(nu_new0-nu)),max(abs(tau_new0-tau))]; % for monitoring convergence

site.ds=ds;
site.type=qc.type;
site.nu1=nu_new;
site.tau1=tau_new;

if isfield(qt,'mz2')
  % site update for transformed variable z2
  
  if length(dfv)>=2
    df=dfv(2);
  else
    df=dfv(1);
  end
  if isfield(qc,'tau_min2')
    tau_min=qc.tau_min2;
  else
    tau_min=[];
  end
  
  nu=qc.nu2;
  tau=qc.tau2;
  tau_new=tau;
  nu_new=nu;
  
  Vzt=qt.Vz2;
  tau_zt=1./Vzt;
  
  % find site terms with proper tilted moments
  ii1=isfinite(Vzt) & Vzt>0 & isfinite(qt.logZ);
  
  % update the site location parameters
  tau_new0=tau;
  tau_new0(ii1)=(tau_zt(ii1)-1./qc.Vz2(ii1))/eta;
  tau_new(ii1)=(1-df)*tau(ii1) +df*tau_new0(ii1); % damped update
  %tau_new(ii1)=(1-df)*tau(ii1) +(df/eta)*(tau_zt(ii1) -1./qc.Vz2(ii1));
  
  if ~isempty(tau_min)
    % constrain the tilted variances if the site precisions get too small: tau < tau_min
    ii2=tau_new<tau_min;
    if any(ii2)
      tau_zt(ii2) = 1./qc.Vz2(ii2) +(eta/df)*(tau_min+(df-1)*tau(ii2));
      tau_new0(ii2) = (tau_zt(ii2)-1./qc.Vz2(ii2))/eta;
      tau_new(ii2)=(1-df)*tau(ii2) +df*tau_new0(ii2);
      %tau_new(ii2)=(1-df)*tau(ii2) +(df/eta)*(tau_zt(ii2) -1./qc.Vz2(ii2));
    end
  end
  
  % update the site location parameters
  nu_new0=nu;
  nu_new0(ii1)=(qt.mz2(ii1).*tau_zt(ii1)-qc.mz2(ii1)./qc.Vz2(ii1))/eta;
  nu_new(ii1)=(1-df)*nu(ii1) +df*nu_new0(ii1);
  %nu_new(ii1)=(1-df)*nu(ii1) +(df/eta)*(qt.mz2(ii1).*tau_zt(ii1) -qc.mz2(ii1)./qc.Vz2(ii1));
  
  ds=[max(abs(nu_new0-nu)),max(abs(tau_new0-tau))]; % for monitoring convergence
  site.ds=[site.ds ds];
  site.nu2=nu_new;
  site.tau2=tau_new;
end

%dtau=zeros(ns,1);
%dnu=zeros(ns,1);
% dtau(ii1) = (1./qt.Vz1(ii1) -1./qc.Vz1(ii1))/eta -qc.tau1(ii1);
% dnu(ii1) = (qt.mz1(ii1)./qt.Vz1(ii1) -qt.mz1(ii1)./qc.Vz1(ii1))/eta -qc.nu1(ii1);
