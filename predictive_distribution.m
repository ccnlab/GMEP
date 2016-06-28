function qp=predictive_distribution(qp,y,lik_type,opt)
%
% Compute E(y), Var(y), and predictive densities p(y) for observations y
% distributed as 
%
% p(y,f,u) = p(y|f,u)*N(u|mu,Vu)*N(f|mf,Vf)
% 
% N(f|mf,Vf) is a Gaussian approximation to the latent function values f=f(x)
% and N(u|mu,Vu) is a Gaussian approximation for optional hyperparameters u
%
% The approximations for f and u are model specific and require separate
% implementations, see e.g. qf_pred in approximation_hglm.m for linear
% models.
%
% Pasi Jylänki 2014

switch lik_type
  case 'Gaussian'
    mf=qp.mf(:);
    Vf=qp.Vf(:);  
    [n1,n2]=size(qp.mf);
    if isfield(qp,'mh')
      % compute E(y), Var(y), p(y) w.r.t.
      % p(y,f,h) ~ N(u|mu,Vu)*N(f|mf,Vf)*N(y|f,exp(h))
      
      mu=qp.mh(:);
      Vu=qp.Vh(:);
      uval=opt.quad_grid;
      quadw=opt.quad_weights;
      su=sqrt(Vu);
      uv=bsxfun(@plus,mu,su*uval);
      du=uv(:,2)-uv(:,1);
      
      qu=exp(-bsxfun(@plus,log(2*pi*Vu)/2, 0.5*uval.^2));
      Vr=((exp(uv).*qu)*quadw).*du;
      
      my=mf;
      Vy=Vf+Vr;
      
      if ~isempty(y)
        y=y(:);
        Vs=bsxfun(@plus,Vf,exp(uv));
        fu = exp( bsxfun(@rdivide, -0.5*(y-mf).^2, Vs) -0.5*log(2*pi*Vs) ...
          -bsxfun(@plus,log(2*pi*Vu)/2, 0.5*uval.^2) );
      
        py=(fu*quadw).*du;
        qp.logpy=reshape(log(py),n1,n2);
      end

    else
      % Compute E(y), Var(y), p(y) w.r.t. p(y,f) ~ N(f|mf,Vf)*N(y|f,h)
      % here noise variance h is known
      h=qp.h(:);
      my=mf;
      Vy=Vf+h;
      
      if ~isempty(y)
        y=y(:);
        Vs=Vf+h;
        logpy = bsxfun(@rdivide, -0.5*(y-mf).^2, Vs) -0.5*log(2*pi*Vs);
        qp.logpy=reshape(logpy,n1,n2);
      end
    end
    qp.my=reshape(my,n1,n2);
    qp.Vy=reshape(Vy,n1,n2);
    
  case 'Laplace'    
    mf=qp.mf(:);
    Vf=qp.Vf(:);  
    [n1,n2]=size(qp.mf);
    if isfield(qp,'mh')
      % compute E(y), Var(y), p(y) w.r.t.
      % p(u,w) ~ N(u|mu,Vu)*N(f|mf,Vf)*La(y|f,lambda)
      % where lambda = exp(u/2-log(2)/2) => 2*lambda^2 = exp(u) = Var(y|lambda)
      
      mu=qp.mh(:);
      Vu=qp.Vh(:);
      uval=opt.quad_grid;
      quadw=opt.quad_weights;
      su=sqrt(Vu);
      uv=bsxfun(@plus,mu,su*uval);
      du=uv(:,2)-uv(:,1);
      
      qu=exp(-bsxfun(@plus,log(2*pi*Vu)/2, 0.5*uval.^2));
      Vr=((exp(uv).*qu)*quadw).*du;
      my=mf;
      Vy=Vf+Vr;
      if ~isempty(y)
        y=y(:);
        
        mr=y-mf;
        sf=sqrt(Vf);
        zf=mr./sf;
        lnl=uv/2-log(2)/2; % log(lambda)
        ilm=exp(-lnl);
        zs=bsxfun(@times,sf,ilm);
        rs=bsxfun(@times,Vf/2,exp(-2*lnl));
        rm=bsxfun(@times,mr,ilm);
        
        Z1=rs+rm +ln_normcdf(bsxfun(@minus,-zs,zf));
        Z2=rs-rm +ln_normcdf(bsxfun(@plus,-zs,zf));
        Z1=exp(Z1);
        Z2=exp(Z2);
        Z=(Z1+Z2).*ilm/2;
        
        fu = Z.*exp(-bsxfun(@plus,log(2*pi*Vu)/2, 0.5*uval.^2) );
        py=(fu*quadw).*du;
        qp.logpy=reshape(log(py),n1,n2);
      end
    else
      % Compute E(y), Var(y), p(y) w.r.t. p(y,f) ~ N(f|mf,Vf)*La(y|f,lambda)
      % here noise variance h is known: h =2*lambda^2 = Var(y|lambda)
      
      h=qp.h(:);
      my=mf;
      Vy=Vf+h;
      
      if ~isempty(y)
        y=y(:);
        
        mr=y-mf;
        sf=sqrt(Vf);
        zf=mr./sf;
        lnl=0.5*log(0.5*h); % log(lambda)
        ilm=exp(-lnl);
        zs=bsxfun(@times,sf,ilm);
        rs=bsxfun(@times,Vf/2,exp(-2*lnl));
        rm=bsxfun(@times,mr,ilm);
        
        Z1=rs+rm +ln_normcdf(bsxfun(@minus,-zs,zf));
        Z2=rs-rm +ln_normcdf(bsxfun(@plus,-zs,zf));
        Z1=exp(Z1);
        Z2=exp(Z2);
        logpy=log(Z1+Z2)-lnl-log(2);
        qp.logpy=reshape(logpy,n1,n2);
        
        %         ci=randi(length(mf));
        %         lambda= exp(lnl(ci));
        %         tol=1e-6;
        %         Zi=quad(@(f) exp(-abs(y(ci)-f)./lambda)/2/lambda .*normpdf(f,mf(ci),sqrt(Vf(ci))),...
        %           mf(ci)-15*sqrt(Vf(ci)),mf(ci)+15*sqrt(Vf(ci)),tol);
        %         [logpy(ci) log(Zi)]
      end
    end
    qp.my=reshape(my,n1,n2);
    qp.Vy=reshape(Vy,n1,n2);
    
  case 'Probit'
    % Compute p(y) w.r.t. p(y,w) = Normcdf(y*w)*N(w|mw,Vw)
    % where y=[-1 1]
    
    mf=qp.mf(:);
    Vf=qp.Vf(:);  
    [n1,n2]=size(qp.mf);
    
    if isempty(y)
      % compute p(y=1)=E(y) for a Bernoulli trial, where y=[0,1]
      % no transformation needed
    else
      % otherwise compute p(y) for the given y
      y=y(:);
      y=2*y-1; % transform [0 1] inputs to [-1 1]
      mf=mf.*y;
      y2=y.^2;
      Vf=Vf.*y2;
    end
    
    a=sqrt(1+Vf);
    z=mf./a;
    logZt=ln_normcdf(z);
    Zt=exp(logZt);
    
    %Nz=exp(-0.5*z.^2)/sqrt(2*pi);
    %mft = mf + Vf.*Nz./Zt./a;
    %Vft = Vf - Vf.^2 .*(Nz./Zt./(1+Vf)).*(z+Nz./Zt);
    
    qp.logpy=reshape(logZt,n1,n2);
    qp.py=reshape(Zt,n1,n2);
  otherwise
    
    fprintf('Unknown likelihood.\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check the predictive moments using quad
if 0
  ci=randi(numel(mf))
  A=[1 0; 0 1; 1 -1]';
  minu=mu(ci)-8*su(ci);
  maxu=mu(ci)+8*su(ci);
  minf=mf(ci)-8*sqrt(Vf(ci));
  maxf=mf(ci)+8*sqrt(Vf(ci));
  miny=mf(ci)-6*(Vf(ci)+exp(mu(ci)+3*su(ci)));
  maxy=mf(ci)+6*(Vf(ci)+exp(mu(ci)+3*su(ci)));
  eta=1;
  
  if isempty(y), 
    yi=0; 
  else
    yi=y(ci);
  end
  
  switch lik_type
    % exact integrand
    case 'Gaussian'
      Cq=0;
      fh=@(y,f,u) normpdf(y,f,exp(0.5*u)).^eta ...
        .*normpdf(f,mf(ci),sqrt(Vf(ci))).*normpdf(u,mu(ci),sqrt(Vu(ci)));
    case 'Laplace'
      Cq=10;
      fla=@(u) u/2-log(2)/2;
      fh=@(y,f,u) exp( eta*(-abs(y-f)./exp(fla(u)) -fla(u)-log(2)) -Cq)...
        .*normpdf(f,mf(ci),sqrt(Vf(ci))).*normpdf(u,mu(ci),sqrt(Vu(ci)));
  end
  
  atol=1e-10;
  rtol=1e-6;
  Zy=quad2d(@(f,u) fh(yi,f,u),minf,maxf,minu,maxu,'AbsTol',atol,'RelTol',rtol);
  Zz=triplequad(@(y,f,u) fh(y,f,u),miny,maxy,minf,maxf,minu,maxu,rtol);
  my=triplequad(@(y,f,u) y.*fh(y,f,u),miny,maxy,minf,maxf,minu,maxu,rtol)/Zz;
  Vy=triplequad(@(y,f,u) y.^2 .*fh(y,f,u),miny,maxy,minf,maxf,minu,maxu,rtol)/Zz - my^2;
  logpy=log(Zy)+Cq;
  disp([qp.logpy(ci) logpy; qp.my(ci) my; qp.Vy(ci) Vy]*A)
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
