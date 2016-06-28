function qt=tilted_moments(qc,y,eta,site_type,opt)
%
% Compute the normalization terms and moments E(w), Var(w), E(u), Var(u) 
% of tilted ditributions of the form
%
% p(z1,z2) ~ N(z1|qc.mz1,qc.Vz1)*N(z2|qc.mz2,qc.Vz2)*p(y|z1,z2)^eta
% 
% N(z1|mz1,Vz1) is a Gaussian cavity approximation to a univariate random variable 
% that is obtained by linearly transforming the model paramters. E.g. with
% a linear model z1=x'*w, where x is an input vector and w the coefficient vector.
% 
% N(z2|mz2,Vz2) is a Gaussian cavity approximation to similarly transformed model
% hyperparameters, e.g., noise variance sigma2 could be modelled as sigma2=exp(z2).
%
% 0<eta<=1 is a fraction paramter that can be used to implement fractiona
% updates by setting eta<1
% 
% The approximations for z1 and z2 are model specific and require separate
% implementations, see e.g. cavity_moments in approximation_hglm.m for linear
% models.
%
% Pasi Jylänki 2014

switch site_type
  case 'Gaussian'
    % Only needed if hyperparameters are present
    % compute moments of p(u,w) ~ N(u|mu,Vu)*N(w|mw,Vw)*N(y|w,exp(u))^eta
    
    mw=qc.mz1;
    Vw=qc.Vz1;
    mu=qc.mz2;
    Vu=qc.Vz2;
    uval=opt.quad_grid;
    quadw=opt.quad_weights;
    
    su=sqrt(Vu);
    uv=bsxfun(@plus,mu,su*uval);
    du=uv(:,2)-uv(:,1);
    
    if ~isempty(y)
      % if y is non-zero, reparameterize as r=y-w and integrated over r:
      % p(u,r) ~ N(u|mu,Vu)*N(r|y-mw,Vw)*N(r|0,exp(u))^eta
      mr=y-mw;
    else
      mr=mw;
    end
    
    Vs=bsxfun(@plus,Vw,exp(uv)/eta);
    %       fu = exp( bsxfun(@rdivide, -0.5*mr.^2, Vs) -0.5*log(2*pi*Vs) ...
    %         +((log(2*pi)+uv)*(1-eta)/2 -log(eta)/2) ...
    %         -bsxfun(@plus,log(2*pi*Vu)/2, 0.5*bsxfun(@rdivide,bsxfun(@minus,uv,mu).^2,Vu)) );
    
    fu = exp( bsxfun(@rdivide, -0.5*mr.^2, Vs) -0.5*log(2*pi*Vs) ...
      +((log(2*pi)+uv)*(1-eta)/2 -log(eta)/2) ...
      -bsxfun(@plus,log(2*pi*Vu)/2, 0.5*uval.^2) );
    
    Zt=(fu*quadw).*du;
    logZt=log(Zt);
    
    tmp=fu.*uv;
    mut=(tmp*quadw).*du./Zt;
    Vut=((tmp.*uv)*quadw).*du./Zt-mut.^2;
    
    % par 1
    %     t=1./(Vw+exp(uv)/eta);
    %     tmp=fu.*t;
    %     Et = ((tmp)*c)/Zt;
    %     Et2 = ((tmp.*t)*c)/Zt;
    %     Vt = Et2-Et^2;
    %
    %     mwt=(1-Et*Vw)*mr
    %     Vwt=Vw-Vw^2*(Et-Vt*mr^2)
    
    % par 2
    g=1./bsxfun(@plus,1./Vw,eta*exp(-uv));
    %g=bsxfun(@times,Vw/eta,exp(uv))./Vs;
    tmp=fu.*g;
    Eg = ((tmp)*quadw).*du./Zt;
    Eg2 = ((tmp.*g)*quadw).*du./Zt;
    Vg = Eg2-Eg.^2;
    
    nu_w=mr./Vw;
    mwt=Eg.*nu_w;
    Vwt=Eg+Vg.*nu_w.^2;
    if ~isempty(y)
      mwt=y-mwt;
    end
    
    qt.logZ=logZt;
    qt.mz1=mwt;
    qt.Vz1=Vwt;
    qt.mz2=mut;
    qt.Vz2=Vut;
    
  case 'Laplace'
    
    mw=qc.mz1;
    Vw=qc.Vz1;
    if ~isempty(y)
      % if y is non-zero, reparameterize as r=y-w and integrated over r:
      % p(u,r) ~ N(u|mu,Vu)*N(r|y-mw,Vw)*La(r|0,exp(u))^eta
      mr=y-mw;
    else
      mr=mw;
    end
    
    if isfield(qc,'mz2')
      % When hyperparameters are present
      % compute moments of p(u,w) ~ N(u|mu,Vu)*N(w|mq,Vq)*La(y|w,lambda)^eta
      % where lambda = exp(u/2-log(2)/2) => 2*lambda^2 = exp(u) = Var(w|lambda)
      
      mu=qc.mz2;
      Vu=qc.Vz2;
      uval=opt.quad_grid;
      quadw=opt.quad_weights;
      su=sqrt(Vu);
      uv=bsxfun(@plus,mu,su*uval);
      du=uv(:,2)-uv(:,1);
      
      % auxililiary variables
      sw=sqrt(Vw);
      zw=mr./sw;
      lnl=uv/2-log(2)/2;
      tmp1=-lnl+log(eta);
      ilm=exp(tmp1);
      zs=bsxfun(@times,sw,ilm);
      rs=bsxfun(@times,Vw/2,exp(2*tmp1)); %0.5*Vw /la^{2} *eta^2
      rm=bsxfun(@times,mr,ilm);
      lnqu=-bsxfun(@plus,log(2*pi*Vu)/2,0.5*uval.^2);
      %lnqu=-log(2*pi*Vu)/2 -0.5*(uv-mu).^2 ./Vu;
      lnrho=-0.5*zw.^2 -0.5*log(2*pi*Vw); %rho=exp(lnrho);
      
      % version 1
      % tilted normalization
      %Z1=rs+rm +ln_normcdf(bsxfun(@minus,-zs,zw));
      %Z2=rs-rm +ln_normcdf(bsxfun(@plus,-zs,zw));
      %Z1=exp(Z1);
      %Z2=exp(Z2);
      %Z=Z1+Z2;
      
      %fu=Z.*exp(-eta*log(2) -eta*lnl +lnqu);
      %tmp2=-eta*log(2) -eta*lnl +lnqu;
      %fu=Z.*exp(tmp2);
      %Zt=(fu*quadw).*du;
      %logZt=log(Zt);
      
      % E(w)
      % %fu2=(Z1-Z2).*exp(-eta*log(2) +log(eta) -(1+eta)*lnl +lnqu);
      %fu2=(Z1-Z2).*exp(tmp2+tmp1); %tmp1=-lnl+log(eta);
      %Ev1=(fu2*quadw).*Vw.*du./Zt;
      %mwt=mr+Ev1;
      
      % Var(w)
      % %fu3=exp(-2*lnl+2*log(eta)).*fu ...
      % % fu3=Z.*exp(-eta*log(2) +2*log(eta) -(2+eta)*lnl +lnqu) ...
      % %   -exp(lnrho +(1-eta)*log(2) +log(eta) -(1+eta)*lnl +lnqu);
      %fu3=Z.*exp(tmp2+2*tmp1) ...
      %  -exp( bsxfun(@plus,tmp2+tmp1,lnrho+log(2)) );
      %Ev2=Vw.^2 .*(fu3*quadw).*du./Zt;
      %Vwt=Vw+Ev2-Ev1.^2;
      
      % version 2 (stabilize with C0)
      % tilted normalization
      tmp2=-eta*log(2) -eta*lnl +lnqu;
      Z1=rs+rm +ln_normcdf(bsxfun(@minus,-zs,zw))+tmp2;
      Z2=rs-rm +ln_normcdf(bsxfun(@plus,-zs,zw))+tmp2;
      C0=max(max(Z1,[],2),max(Z2,[],2));
      Z1=exp(bsxfun(@minus,Z1,C0));
      Z2=exp(bsxfun(@minus,Z2,C0));
      Z=Z1+Z2;
      
      fu=Z;
      Zt=(fu*quadw).*du;
      logZt=log(Zt)+C0;
      
      % E(u) and Var(u)
      tmp=fu.*uv;
      mut=(tmp*quadw).*du./Zt;
      Vut=((tmp.*uv)*quadw).*du./Zt-mut.^2;
      
      % E(w)
      fu2=(Z1-Z2).*exp(tmp1); %tmp1=-lnl+log(eta);
      Ev1=(fu2*quadw).*Vw.*du./Zt;
      mwt=mr+Ev1;
      
      %Var(w)
      fu3=Z.*exp(2*tmp1)-exp( bsxfun(@plus,tmp2+tmp1,lnrho+log(2)-C0) );
      Ev2=Vw.^2 .*(fu3*quadw).*du./Zt;
      Vwt=Vw+Ev2-Ev1.^2;
      
      qt.mz2=mut;
      qt.Vz2=Vut;
    else
      % In this case hyperparameters are fixed to v
      % moments of p(w) ~ N(w|mq,Vq)*La(y|w,lambda)^eta
      % where v =2*lambda^2 = Var(w|lambda)
      
      v=qc.v; % v =2*lambda^2 = Var(w|lambda)
      sw=sqrt(Vw);
      zw=mr./sw;
      lnl=0.5*log(0.5*v); % log(lambda)
      
      ilm=exp(-lnl+log(eta));
      zs=bsxfun(@times,sw,ilm);
      rs=bsxfun(@times, Vw/2, exp(2*(-lnl+log(eta))) ); %Vw /la^{2} *eta^2
      rm=bsxfun(@times,mr,ilm);
      
      Z1=rs+rm +ln_normcdf(bsxfun(@minus,-zs,zw));
      Z2=rs-rm +ln_normcdf(bsxfun(@plus,-zs,zw));
      %Z1=exp(Z1);
      %Z2=exp(Z2);
      C0=max(Z1,Z2);
      Z1=exp(Z1-C0);
      Z2=exp(Z2-C0);
      Z=Z1+Z2;
      
      % normalization
      tmp1=-eta*log(2)-eta*lnl;
      %logZt=log(Z)+tmp1;
      %Zt=Z.*exp(tmp1);
      logZt=log(Z)+tmp1+C0; 
      %Zt=Z*exp(tmp1+C0);
      
      % E(w)
      %Ev1=(Z1-Z2).*exp(tmp1 +log(eta) -lnl).*Vw./Zt;
      Ev1=(Z1-Z2).*exp(log(eta)-lnl).*Vw./Z;
      mwt=mr+Ev1;
      
      %Var(w)
      lnrho=-0.5*zw.^2 -0.5*log(2*pi*Vw); %rho=exp(lnrho);
      %Ev2=Z.*exp(tmp1 +2*log(eta) -2*lnl) ...
      %  -exp( bsxfun(@plus,tmp1-lnl,lnrho +log(2) +log(eta)) );
      %Ev2=Vw.^2 .*Ev2./Zt;
      Ev2=exp(2*log(eta)-2*lnl)-exp(-lnl+lnrho+log(2)+log(eta)-C0)./Z;
      Ev2=Vw.^2 .*Ev2;
      Vwt=Vw+Ev2-Ev1.^2;
    end
    
    if ~isempty(y)
      mwt=y-mwt;
    end
    qt.logZ=logZt;
    qt.mz1=mwt;
    qt.Vz1=Vwt;
  case 'Probit'
    % Compute the moments of p(w) ~ Normcdf(y*w)*N(w|mw,Vw)
    % note that fractional updates are not implemented eta=1
    % no hyper parameters
    
    mw=qc.mz1;
    Vw=qc.Vz1;
    if ~isempty(y)
      % transform into p(f) ~ Normcdf(f)*N(f|y.*mw,y.^2 .*Vw), where f=y.*w
      mf=mw.*y;
      y2=y.^2;
      Vf=Vw.*y2; 
    else
      mf=mw;
      Vf=Vw;
    end
    
    % tilted moments
    a=sqrt(1+Vf);
    z=mf./a;
    logZt=ln_normcdf(z);
    Zt=exp(logZt);
    Nz=exp(-0.5*z.^2)/sqrt(2*pi);
    
    mft = mf + Vf.*Nz./Zt./a;
    Vft = Vf - Vf.^2 .*(Nz./Zt./(1+Vf)).*(z+Nz./Zt);
    
    if ~isempty(y)
      % transform back to moments of w
      mwt=mft./y;
      Vwt=Vft./y2; 
    end
    qt.logZ=logZt;
    qt.mz1=mwt;
    qt.Vz1=Vwt;
    
  otherwise
    fprintf('Unknown site function.\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check the analytical tilted moments using quad
if 0
  if isfield(qc,'mz2')
    ci=randi(numel(mw))
    %A=[1 0; 0 1; 1 -1]';
    minu=mu(ci)-8*su(ci);
    maxu=mu(ci)+8*su(ci);
    minw=mw(ci)-8*sqrt(Vw(ci));
    maxw=mw(ci)+8*sqrt(Vw(ci));
    if isempty(y), y(ci)=0; end
    
    switch site_type
      % exact integrand
      case 'Gaussian'
        Cq=0;
        fh=@(w,u) normpdf(y(ci),w,exp(0.5*u)).^eta ...
          .*normpdf(w,mw(ci),sqrt(Vw(ci))).*normpdf(u,mu(ci),sqrt(Vu(ci)));
      case 'Laplace'
        Cq=10;
        fla=@(u) u/2-log(2)/2;
        %   fh=@(w,u) exp( eta*(-abs(w)./exp(fla(u)) -fla(u)-log(2))  ...
        %     -0.5*(w-mw).^2 /Vw -0.5*log(2*pi*Vw) -0.5*(w-mu).^2 /Vu -0.5*log(2*pi*Vu) -C);
        fh=@(w,u) exp( eta*(-abs(y(ci)-w)./exp(fla(u)) -fla(u)-log(2)) -Cq)...
          .*normpdf(w,mw(ci),sqrt(Vw(ci))).*normpdf(u,mu(ci),sqrt(Vu(ci)));
        % fh_l=@(w,u) (exp(-abs(w)./exp(fla(u)))./(2*exp(fla(u)))).^eta;
    end
    
    atol=1e-10;
    rtol=1e-6;
    Zt2=quad2d(@(w,u) fh(w,u),minw,maxw,minu,maxu,'AbsTol',atol,'RelTol',rtol);
    mut2=quad2d(@(w,u) u.*fh(w,u),minw,maxw,minu,maxu,'AbsTol',atol,'RelTol',rtol)/Zt2;
    Vut2=quad2d(@(w,u) u.^2 .*fh(w,u),minw,maxw,minu,maxu,'AbsTol',atol,'RelTol',rtol)/Zt2 - mut2^2;
    mwt2=quad2d(@(w,u) w.*fh(w,u),minw,maxw,minu,maxu,'AbsTol',atol,'RelTol',rtol)/Zt2;
    Vwt2=quad2d(@(w,u) w.^2 .*fh(w,u),minw,maxw,minu,maxu,'AbsTol',atol,'RelTol',rtol)/Zt2 -mwt2^2;
    logZt2=log(Zt2)+Cq;
    
    disp([logZt(ci) logZt2 logZt(ci)-logZt2; mut(ci) mut2 mut(ci)-mut2; ...
      Vut(ci) Vut2 Vut(ci)-Vut2; mwt(ci) mwt2 mwt(ci)-mwt2; Vwt(ci) Vwt2 Vwt(ci)-Vwt2])
    Cwu=quad2d(@(w,u) w.*u.*fh(w,u),minw,maxw,minu,maxu)/Zt2-mwt2*mut2;
    Cwu=Cwu/sqrt(Vwt2)/sqrt(Vut2)
  else
    % 1-d integrals
    tol=1e-9;
    ci=randi(length(mw));
    wmin=mw(ci)-10*sqrt(Vw(ci));
    wmax=mw(ci)+10*sqrt(Vw(ci));
    switch site_type
      case 'Laplace'
        lambda= exp(lnl(ci));
        fh=@(f) exp(-eta*abs(y(ci)-f)./lambda -eta*log(2*lambda)) .*normpdf(f,mw(ci),sqrt(Vw(ci)));
      case 'Probit'
        fh=@(w) normcdf(y(ci).*w).*normpdf(w,mw(ci),sqrt(Vw(ci)));
    end
    Zi=quad(@(w) fh(w),wmin,wmax,tol);
    mi=quad(@(w) w.*fh(w),wmin,wmax,tol)/Zi;
    Vi=quad(@(w) w.^2 .*fh(w),wmin,wmax,tol)/Zi-mi^2;
    [logZt(ci) log(Zi) logZt(ci)-log(Zi); mwt(ci) mi mwt(ci)-mi; Vwt(ci) Vi Vwt(ci)-Vi]
  end
end

end
  
function y=ln_normcdf(x)
  y=x;
  ii1=x>-6.5;
  %y = log(0.5*erf(x/sqrt(2)) + 0.5);
  y(ii1) = log(0.5*erfc(-x(ii1)./sqrt(2)));
  
  ii1=~ii1;
  if any(any(ii1))
    z=x(ii1).^2;
    y(ii1) = -0.5*log(2*pi)-0.5*z-log(-x(ii1));
    z=1./z;
    y(ii1) = y(ii1) +log(1-z.*(1-3*z.*(1-5*z.*(1-7*z.*(1-9*z.*(1-11*z.*(1-13*z)))))));
  end
end
