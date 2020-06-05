      program breakage_main 
      implicit none
	integer, parameter :: dp  = kind(1.0d0), t=10000_dp, NSTR=6,
     *                  	  Ndirect=3,Nshear=3
	real(kind=dp), dimension(NSTR) :: eps_pls, Strain, Stress,delta,
     *dptri_dsigmatri,dqtri_dsigmatri,Ttri,H1,dystar_dsigmatri,strainInc
     *,Dev_strain,stress_tri_Inc,stress_tri,vector_smean,dev_stress_tri,
     *deps_pls,dstress,dev_stress,Dev_eps_pls,d_dev_stress,e_stress,
     *stress_tri2,e_e,e_strain
	real(kind=dp), dimension(NSTR,NSTR) ::Matrix_P,D,Matrix_Pinv,Iu,Iv
 	real(kind=dp) ::E,K, Ec, G, M,xnu,theta,smean,alamda,i,B,p,q,k1,
     *k2,k3,ptri,qtri,Ebtri,ytri,dytri_dEbtri,dEbtri_dptri,dEbtri_dqtri,
     *dytri_dptri,dytri_dqtri,dEbtri_dB,dystar_dB,dytri_dB,eps_ev,
     *eps_es,dystar_dEbtri,dystar_dptri,dystar_dqtri,Utri,dL,dB,eps_v,
     *deps_v,eps_S,deps_s,eps_ps,eps_pv,H2,H3,H4,Pc,skk,skk_pls,d_p,
     *d_q,Eb,dEb,dsmean,EEbtri,dif,eps_ev_tri,eps_es_tri,omega,pi,
     *e0,emax,en0,de,n,n0,nmax,dn,eps_v0,nchange,rad,deg,q2
c***************************************************************************
c Specify material properties
      K = 4608.0                 ! Itai's undrained parameters
      G = 4710.0                 ! Itai's undrained parameters
c      Ec = 50.0_dp                   ! Itai's undrained parameters
      M = 1.5                     ! Itai's undrained parameters
      theta=0.9
c 	Ec=3.53
c	Pc=sqrt(2.0_dp*K*Ec/theta)     ! Itai's undrained parameters 
c  	K=4608.0_dp
c  	G=4710.0_dp
c   	M=1.5_dp
c Determine critical compression pressure pc from experiment
      Pc=190 
      Ec=theta*Pc**2.0_dp/(2.0_dp*K)
c
	xnu=(3.0_dp*K-2.0_dp*G)/(6.0_dp*K+2.0_dp*G)
	E=2.0_dp*G*(1.0_dp+xnu)
	alamda=2.0_dp*G*(E-2.0_dp*G)/(6.0_dp*G-2.0_dp*E)
	pi=3.141592653589793_dp
	rad = pi / 180.0_dp        ! degrees to radians
	deg = 180.0_dp / pi        ! radians to degrees
	omega=45.0_dp*rad         
	e0=1.5_dp                  ! e: void ratio
	emax=1.8_dp
c
	en0=(e0+emax)/2.0_dp       ! n: porosity
c	eps_v0=(emax-e0)/(1.0_dp+en0)
	n0=e0/(1.0_dp+e0)
	nmax=emax/(1.0_dp+emax)
	nchange=0.0_dp

c
c Initialization
	e=e0
	n=n0
	nchange=0             ! poro change
c
c  Define kroneck Matrix and after vectorization
       delta(1:Ndirect)=1.0_dp
	 delta(4)=0.0_dp
	 delta(5)=0.0_dp
	 delta(6)=0.0_dp	
c
c  Define transfer matrix appropriate for engineering shears
      Do K1=1,Ndirect
	  Matrix_P(k1,k1)=1.0_dp
	  Matrix_Pinv(k1,k1) = 1.0_dp
	 End do
	Do k2=Ndirect+1, NSTR
	  Matrix_P(k2,k2)=2.0_dp
	  Matrix_Pinv(k2,k2) = 0.5_dp
	End do
c		    
c  Define elastic modulus matrix D
      D(1:NSTR, 1:NSTR) = 0.0_dp
	do k1=1,Ndirect
	   do k2=1,Ndirect
	     D(k2,k1)=aLamda
	   end do
	  D(k1,k1)=2.0_dp*G+aLamda
	end do
	do k3=Ndirect+1,NSTR
      D(k3,k3)=2.0_dp*G
	end do

c********************************************************************
       do 100 i=1,t
c
c        open(10,file ='negative_strains_inp_iso.dat',
c     *	   action='read')
        open(10,file ='ABA_INP.dat',
     *	   action='read')
        open(6,file ='output.log', action='write')
c	  open(7,file ='e&p&q_out.log', action='write')
c	  open(8,file ='n&p&q_out.log', action='write')
c	  open(9,file ='nchange&p&q_out.log', action='write')	  
c	  open(11,file ='stn(1)&str(1)_out.log', action='write')
c	  open(12,file ='eps_v&p_out.log', action='write')
c	  open(13,file ='B_y.log', action='write')
c	  open(14,file ='eps_s&q_out.log', action='write')
c	  open(15,file ='difptri&dif_out.log', action='write')
c	  open(16,file ='stn(1)&strtri(1)_out.log', action='write')
c	  open(17,file ='eps_pv&eps_ev.log', action='write')
c	  open(18,file ='stn(4)&str(4)_out.log', action='write')
c	  open(19,file ='dev_stress_tri1to3_out.log', action='write')
c	  open(20,file ='dev_stress_tri4to6_out.log', action='write')
c	  open(21,file ='dqtri_dsigmatri_out.log', action='write')
c	  open(22,file ='Ebtri&EEbtri_out.log', action='write')
c	  open(23,file ='theta_p_q.log', action='write')
c	  open(24,file ='eps_s_q_Pc.log', action='write')
c	  open(25,file ='q_Pc_B.log', action='write')
c
c  Input engineering strains
	  read (10,*) strainInc(1),strainInc(2),strainInc(3),
     *	          strainInc(4),strainInc(5),strainInc(6)
c      strainInc(1:Ndirect)=5.0d-5
c	strainInc(Ndirect+1:NSTR)=0.0_dp  
c
	strain(1:NSTR) = strain(1:NSTR) + strainInc(1:NSTR)
	deps_v=-sum(strainInc(1:Ndirect)) 
	Dev_strain(1:NSTR) = strain(1:NSTR)
     *                 	 -sum(Strain(1:Ndirect))*delta(1:NSTR)/3.0_dp 
	eps_v = -sum(strain(1:Ndirect)) !!!!  to transfer FEM convention of sign into soil mechanics(triaxial space)
      eps_s = sqrt((2.0_dp/3.0_dp)*dot_product(Dev_strain(1:NSTR),
     *	    matmul(Matrix_Pinv(1:NSTR,1:NSTR),Dev_strain(1:NSTR))))
c
	eps_ev_tri=eps_v-eps_pv
	eps_ev_tri=eps_ev+deps_v
	eps_es_tri=eps_s-eps_ps
      e_strain=strain-eps_pls
	e_e=(strain-sum(strain(1:Ndirect))/3*delta)
     *	-(eps_pls-sum(eps_pls(1:Ndirect))/3*delta)
c
c  Caculate elastic trial stress
      stress_tri_Inc(1:NSTR)=(1.0_dp-theta*B)*matmul(D,
     *	matmul(Matrix_Pinv(1:NSTR,1:NSTR),-strainInc(1:NSTR))) 
c	stress_tri(1:NSTR)=stress(1:NSTR)+stress_tri_Inc(1:NSTR)
c
      stress_tri=(1.0-theta*B)*(K*eps_ev_tri*delta 
     *	+2*G*matmul(Matrix_Pinv,e_e))
c
c      stress_tri(1:NSTR)=(1.0_dp-theta*B)*matmul(D,
c     *	matmul(Matrix_Pinv(1:NSTR,1:NSTR),e_strain(1:NSTR))) !verify stress_tri_Inc 
c
	smean = sum(Stress_tri(1:Ndirect))   
      dev_stress_tri(1:NSTR)=stress_tri(1:NSTR)
     *  	                  -smean*delta(1:NSTR)/3.0_dp	  
c         
      EEbtri=theta*(K*eps_ev_tri**2.0_dp/2.0_dp
     *   	 +3.0_dp*G*eps_es_tri**2.0_dp/2.0_dp)
      ptri = sum(Stress_tri(1:Ndirect))/3.0_dp !!
c  or ptri=(1.0_dp-theta*B)*K*eps_ev_tri
	qtri = sqrt((3.0_dp/2.0_dp)*dot_product(Dev_stress_tri(1:NSTR),
     *	     matmul(Matrix_P(1:NSTR,1:NSTR),Dev_stress_tri(1:NSTR))))
 	Ebtri=theta*(ptri**2/K+qtri**2/(3.0_dp*G))
     *	   /(2.0_dp*(1-B*theta)**2)
	ytri=(1.0_dp-B)**2.0_dp*Ebtri/Ec
     *     +(qtri/(M*ptri))**2.0_dp-1.0_dp
c********************************************************************     
c Determine if actively yielding
    	 if (ytri.LT.0) then
c Update elastically
      p=ptri
      q=qtri
	eps_ev=eps_ev_tri
	eps_es=eps_es_tri
      stress=stress_tri
	Eb=Ebtri 
c	Eps_pls=0.0_dp   ! Consider unloading condition, eps_pls=last eps_pls in Pls update, so not 0
c
c	eps_pv=eps_pv
c	eps_ps=eps_ps
c	deps_v=eps_v-eps_v0
c	de=-deps_v*(1.0_dp+e)
c      dn=-deps_v/(1.0_dp-n0)
c      e=e+de  
c	n=n+dn  
c	nchange=nchange-(1.0_dp+n)*deps_v/(1.0_dp+n0) !?poro change?
c********************************************************************
       elseif (ytri.GE.0) then
c  Active plasticity
      dytri_dEbtri=((1.0_dp-B)**2.0_dp)/Ec
	dEbtri_dB=(theta**2.0_dp*(ptri**2.0_dp/K+qtri**2.0_dp/(3.0_dp*G)))
     *  	      /((1.0_dp-theta*B)**3.0_dp)
	dEbtri_dptri=(ptri*theta)/(K*(1.0_dp-theta*B)**2.0_dp)
	dEbtri_dqtri=(qtri*theta)/(3.0_dp*G*(1.0_dp-theta*B)**2.0_dp)
	dptri_dsigmatri=delta/3.0_dp  
c
	if (qtri==0.0_dp) then
	dqtri_dsigmatri=0.0_dp
	else
	dqtri_dsigmatri=1.5_dp*
     *	            matmul(Matrix_P,dev_stress_tri(1:NSTR))/qtri
      end if
c
	dytri_dptri=(-2.0_dp*qtri**2.0_dp)/(M**2.0_dp*ptri**3.0_dp)
	dytri_dqtri=(2.0_dp*qtri)/((M*ptri)**2.0_dp)
	dytri_dB=(-2.0_dp*Ebtri*(1.0_dp-B))/Ec
	dystar_dEbtri=2.0_dp*((1.0_dp-B)**2.0_dp)*(cos(omega))**2/Ec
	dystar_dptri=2.0_dp*((1.0_dp-B)**2.0_dp*Ebtri)*(sin(omega))**2
     *        	/(ptri*Ec)
	dystar_dqtri=(2.0_dp*qtri)/((M*ptri)**2.0_dp)	
c
      dystar_dsigmatri(1:NSTR)=dystar_dptri*dptri_dsigmatri
     *	                    +dystar_dqtri*dqtri_dsigmatri
c
c Obtain Ttrial and Utrial
      Ttri=(dytri_dEbtri*dEbtri_dptri+dytri_dptri)*dptri_dsigmatri
     *    +(dytri_dEbtri*dEbtri_dqtri+dytri_dqtri)*dqtri_dsigmatri
c 
	H1=dystar_dsigmatri
	H2=dot_product(Ttri,(1.0_dp-theta*B)*matmul(D,H1))
	H3=dytri_dB+dytri_dEbtri*dEbtri_dB
	H4=(theta/(1.0_dp-theta*B))
     *  *dot_product(Ttri(1:NSTR),stress_tri(1:NSTR))
c
	Utri=H2-(H3-H4)*dystar_dEbtri
      dL=ytri/Utri
c 
c  Update plastic/breakage stress and strain increments
      dB=dL*dystar_dEbtri  !!!
	deps_pls(1:NSTR)=dL*dystar_dsigmatri(1:NSTR)
      dstress(1:NSTR)=-(1.0_dp-theta*B)*matmul(D,deps_pls)
     *	            -theta*(stress_tri/(1.0_dp-theta*B))*dB
     	eps_pls(1:NSTR)=Eps_pls(1:NSTR)+deps_pls(1:NSTR) 
      skk_pls=sum(eps_pls(1:Ndirect))
	Dev_eps_pls(1:NSTR)=eps_pls(1:NSTR)-skk_pls*delta(1:NSTR)/3.0_dp 
	eps_pv = sum(eps_pls(1:Ndirect)) 
	eps_ps = sqrt((2.0_dp/3.0_dp)*dot_product(Dev_eps_pls(1:NSTR),
     *	     matmul(Matrix_Pinv(1:NSTR,1:NSTR),Dev_eps_pls(1:NSTR))))
c
	eps_ev=eps_v-eps_pv
	eps_es=eps_s-eps_ps	  
c      
	stress(1:NSTR)=stress_tri(1:NSTR)+dstress(1:NSTR)	
      B=B+dB
c
      d_p = sum(dstress(1:Ndirect))/3.0_dp  !!
      d_dev_stress(1:NSTR)=dstress(1:NSTR)
     *  	              -sum(dstress(1:Ndirect))*delta/3.0_dp 
      d_q = sqrt((3.0_dp/2.0_dp)*dot_product(d_dev_stress(1:NSTR)
     *	   ,matmul(Matrix_P,d_dev_stress(1:NSTR))))
c
      p=sum(stress(1:Ndirect))/3.0_dp
	e_stress(1:NSTR)=stress(1:NSTR)
     *	             -sum(stress(1:Ndirect))/3.0_dp*delta(1:NSTR) !deviatoric stresses
      q = sqrt(1.5*dot_product(e_stress(1:NSTR)
     *	   ,matmul(Matrix_P,e_stress(1:NSTR))))	
c
c
c
	dEb=theta*(p*d_p/K+q*d_q/(3.0_dp*G))/(1.0_dp-theta*B)**2.0_dp
     *   +theta**2.0_dp*(p**2.0_dp/K+q**2.0_dp/(3.0_dp*G))
     *   *dB/(1.0_dp-theta*B)
	Eb=Ebtri+dEb
c
c	deps_v=eps_v-eps_v0
c	de=-deps_v*(1.0_dp+e)
c      dn=-deps_v/(1.0_dp-n0)
c	n=n+dn
c      e=e+de
c	nchange=nchange-(1.0_dp+n)*deps_v/(1.0_dp+n0)
	end if
c 
      if (i==1.or.mod(i,1.0)==0.or.i==t) then
      write(6,*) i,B,p,q,eps_pv,eps_ev,eps_v,eps_es, eps_s, eps_ps	     
c	write(7,*) e,p,q 
c	write(8,*) n,p,q
c	write(9,*) nchange,p,q
c	write(11,*) i,strain(1),stress(1) 
c	write(12,*) i,eps_v, p
c	write(13,*) i,B,ytri
c	write(14,*) i,eps_s,q
c	write(16,*) i, strain(1), stress_tri(1)
c	write(17,*) B,eps_pv,eps_ev
c     write(18,*) strain(4),stress(4)
c	write(19,*) dev_stress_tri(1),dev_stress_tri(2),dev_stress_tri(3)
c     write(20,*) dev_stress_tri(4),dev_stress_tri(5),dev_stress_tri(6)
c	write(21,*) dqtri_dsigmatri
c      write(22,*) Ebtri,EEbtri,ytri
c	write(23,*) theta,p,q
c	write(24,*) eps_s,q,Pc
c	write(25,*) q,Pc,B
	end if
c
100   continue
      end

