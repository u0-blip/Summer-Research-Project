def breakage_main()
	integer, parameter: : dp  = kind(1.0), t = 10000, NSTR = 6,
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

# Specify material properties
    K = 4608.0                 #   Itai's undrained parameters
    G = 4710.0                 #   Itai's undrained parameters
#   Ec = 50.0                   #   Itai's undrained parameters
    M = 1.5                     #   Itai's undrained parameters
    theta=0.9
# 	Ec=3.53
#   Pc=sqrt(2.0*K*Ec/theta)     #   Itai's undrained parameters 
#  	K=4608.0
#  	G=4710.0
#   M=1.5
# Determine critical compression pressure p# from experiment
    Pc=190 
    Ec=theta*Pc**2.0/(2.0*K)
	xnu=(3.0*K-2.0*G)/(6.0*K+2.0*G)
	E=2.0*G*(1.0+xnu)
	alamda=2.0*G*(E-2.0*G)/(6.0*G-2.0*E)
	pi=3.141592653589793
	rad = pi / 180.0        #   degrees to radians
	deg = 180.0 / pi        #   radians to degrees
	omega=45.0*rad         
	e0=1.5                  #   e: void ratio
	emax=1.8
	en0=(e0+emax)/2.0       #   n: porosity
#	eps_v0=(emax-e0)/(1.0+en0)
	n0=e0/(1.0+e0)
	nmax=emax/(1.0+emax)
	nchange=0.0

# Initialization
	e=e0
	n=n0
	nchange=0             #   poro change
#  Define kroneck Matrix and after vectorization
    delta(1:Ndirect)=1.0
    delta(4)=0.0
    delta(5)=0.0
    delta(6)=0.0	
#  Define transfer matrix appropriate for engineering shears
    for k1 in range(1,Ndirect):
        Matrix_P(k1,k1)=1.0
        Matrix_Pinv(k1,k1) = 1.0
	for k2 in range(Ndirect+1, NSTR):
        Matrix_P(k2,k2)=2.0
        Matrix_Pinv(k2,k2) = 0.5
#		    
#  Define elastic modulus matrix D
    D(1:NSTR, 1:NSTR) = 0.0
	for k1 in range(1,Ndirect):
	   for k2 in range(1,Ndirect):
	        D(k2,k1)=aLamda
	    D(k1,k1)=2.0*G+aLamda
	for k3 in range(Ndirect+1,NSTR):
        D(k3,k3)=2.0*G

####################
       do 100 i=1,t

#        open(10,file ='negative_strains_inp_iso.dat',
#     *	   action='read')
        open(10,file ='ABA_INP.dat',
     *	   action='read')
        open(6,file ='output.log', action='write')
#    open(7,file ='e&p&q_out.log', action='write')
#    open(8,file ='n&p&q_out.log', action='write')
#    open(9,file ='nchange&p&q_out.log', action='write')	  
#    open(11,file ='stn(1)&str(1)_out.log', action='write')
#    open(12,file ='eps_v&p_out.log', action='write')
#    open(13,file ='B_y.log', action='write')
#    open(14,file ='eps_s&q_out.log', action='write')
#    open(15,file ='difptri&dif_out.log', action='write')
#    open(16,file ='stn(1)&strtri(1)_out.log', action='write')
#    open(17,file ='eps_pv&eps_ev.log', action='write')
#    open(18,file ='stn(4)&str(4)_out.log', action='write')
#    open(19,file ='dev_stress_tri1to3_out.log', action='write')
#    open(20,file ='dev_stress_tri4to6_out.log', action='write')
#    open(21,file ='dqtri_dsigmatri_out.log', action='write')
#    open(22,file ='Ebtri&EEbtri_out.log', action='write')
#    open(23,file ='theta_p_q.log', action='write')
#    open(24,file ='eps_s_q_Pc.log', action='write')
#    open(25,file ='q_Pc_B.log', action='write')
#  Input engineering strains
	  read (10,*) strainInc(1),strainInc(2),strainInc(3),
     *	          strainInc(4),strainInc(5),strainInc(6)
#      strainInc(1:Ndirect)=5.0d-5
#	strainInc(Ndirect+1:NSTR)=0.0  
	strain(1:NSTR) = strain(1:NSTR) + strainInc(1:NSTR)
	deps_v=-sum(strainInc(1:Ndirect)) 
	Dev_strain(1:NSTR) = strain(1:NSTR)
     *                 	 -sum(Strain(1:Ndirect))*delta(1:NSTR)/3.0 
	eps_v = -sum(strain(1:Ndirect)) !!!#    to transfer FEM convention of sign into soil mechanics(triaxial space)
      eps_s = sqrt((2.0/3.0)*dot_product(Dev_strain(1:NSTR),
     *	    matmul(Matrix_Pinv(1:NSTR,1:NSTR),Dev_strain(1:NSTR))))
	eps_ev_tri=eps_v-eps_pv
	eps_ev_tri=eps_ev+deps_v
	eps_es_tri=eps_s-eps_ps
      e_strain=strain-eps_pls
	e_e=(strain-sum(strain(1:Ndirect))/3*delta)
     *	-(eps_pls-sum(eps_pls(1:Ndirect))/3*delta)
#  Caculate elasti# trial stress
      stress_tri_Inc(1:NSTR)=(1.0-theta*B)*matmul(D,
     *	matmul(Matrix_Pinv(1:NSTR,1:NSTR),-strainInc(1:NSTR))) 
#	stress_tri(1:NSTR)=stress(1:NSTR)+stress_tri_Inc(1:NSTR)
      stress_tri=(1.0-theta*B)*(K*eps_ev_tri*delta 
     *	+2*G*matmul(Matrix_Pinv,e_e))
#      stress_tri(1:NSTR)=(1.0-theta*B)*matmul(D,
#     *	matmul(Matrix_Pinv(1:NSTR,1:NSTR),e_strain(1:NSTR))) !verify stress_tri_In# 
	smean = sum(Stress_tri(1:Ndirect))   
      dev_stress_tri(1:NSTR)=stress_tri(1:NSTR)
     *  	                  -smean*delta(1:NSTR)/3.0	  
#         
      EEbtri=theta*(K*eps_ev_tri**2.0/2.0
     *   	 +3.0*G*eps_es_tri**2.0/2.0)
      ptri = sum(Stress_tri(1:Ndirect))/3.0 !!
#  or ptri=(1.0-theta*B)*K*eps_ev_tri
	qtri = sqrt((3.0/2.0)*dot_product(Dev_stress_tri(1:NSTR),
     *	     matmul(Matrix_P(1:NSTR,1:NSTR),Dev_stress_tri(1:NSTR))))
 	Ebtri=theta*(ptri**2/K+qtri**2/(3.0*G))
     *	   /(2.0*(1-B*theta)**2)
	ytri=(1.0-B)**2.0*Ebtri/E     *     +(qtri/(M*ptri))**2.0-1.0
#################################  
# Determine if actively yielding
    if (ytri.LT.0):
# Update elastically
        p=ptri
        q=qtri
	    eps_ev=eps_ev_tri
	    eps_es=eps_es_tri
        stress=stress_tri
	    Eb=Ebtri 
#   Eps_pls=0.0   #   Consider unloading condition, eps_pls=last eps_pls in Pls update, so not 0
#   eps_pv=eps_pv
#   eps_ps=eps_ps
#   deps_v=eps_v-eps_v0
#   de=-deps_v*(1.0+e)
#      dn=-deps_v/(1.0-n0)
#      e=e+de  
#   n=n+dn  
#   nchange=nchange-(1.0+n)*deps_v/(1.0+n0) !?poro change?
###############################
    elif (ytri.GE.0):
#  Active plasticity
        dytri_dEbtri=((1.0-B)**2.0)/E	dEbtri_dB=(theta**2.0*(ptri**2.0/K+qtri**2.0/(3.0*G)))
        */((1.0-theta*B)**3.0)
	    dEbtritri=(ptri*theta)/(K*(1.0-theta*B)**2.0)
	    dEbtri_dqtri=(qtri*theta)/(3.0*G*(1.0-theta*B)**2.0)
	    dptri_dsigmatri=delta/3.0  
        if (qtri==0.0):
            dqtri_dsigmatri=0.0
        else:
            dqtri_dsigmatri=1.5**matmul(Matrix_P,dev_stress_tri(1:NSTR))/qtri
        dytritri=(-2.0*qtri**2.0)/(M**2.0*ptri**3.0)
        dytri_dqtri=(2.0*qtri)/((M*ptri)**2.0)
        dytri_dB=(-2.0*Ebtri*(1.0-B))/E	dystar_dEbtri=2.0*((1.0-B)**2.0)*(cos(omega))**2/E	dystartri=2.0*((1.0-B)**2.0*Ebtri)*(sin(omega))**2
        *        	/(ptri*Ec)
        dystar_dqtri=(2.0*qtri)/((M*ptri)**2.0)	
        dystar_dsigmatri(1:NSTR)=dystartri*dptri_dsigmatri
        *	                    +dystar_dqtri*dqtri_dsigmatri
    # Obtain Ttrial and Utrial
        Ttri=(dytri_dEbtri*dEbtritri+dytritri)*dptri_dsigmatri
        *    +(dytri_dEbtri*dEbtri_dqtri+dytri_dqtri)*dqtri_dsigmatri
    # 
        H1=dystar_dsigmatri
        H2=dot_product(Ttri,(1.0-theta*B)*matmul(D,H1))
        H3=dytri_dB+dytri_dEbtri*dEbtri_dB
        H4=(theta/(1.0-theta*B))
        *  *dot_product(Ttri(1:NSTR),stress_tri(1:NSTR))
        Utri=H2-(H3-H4)*dystar_dEbtri
        dL=ytri/Utri
    # 
    #  Update plastic/breakage stress and strain increments
        dB=dL*dystar_dEbtri  !!!
        deps_pls(1:NSTR)=dL*dystar_dsigmatri(1:NSTR)
        dstress(1:NSTR)=-(1.0-theta*B)*matmul(D,deps_pls)
        *	            -theta*(stress_tri/(1.0-theta*B))*dB
            eps_pls(1:NSTR)=Eps_pls(1:NSTR)+deps_pls(1:NSTR) 
        skk_pls=sum(eps_pls(1:Ndirect))
        Dev_eps_pls(1:NSTR)=eps_pls(1:NSTR)-skk_pls*delta(1:NSTR)/3.0 
        eps_pv = sum(eps_pls(1:Ndirect)) 
        eps_ps = sqrt((2.0/3.0)*dot_product(Dev_eps_pls(1:NSTR),
        *	     matmul(Matrix_Pinv(1:NSTR,1:NSTR),Dev_eps_pls(1:NSTR))))
        eps_ev=eps_v-eps_pv
        eps_es=eps_s-eps_ps	  
    #      
        stress(1:NSTR)=stress_tri(1:NSTR)+dstress(1:NSTR)	
        B=B+dB
        d_p = sum(dstress(1:Ndirect))/3.0  !!
        d_dev_stress(1:NSTR)=dstress(1:NSTR)
        *  	              -sum(dstress(1:Ndirect))*delta/3.0 
        d_q = sqrt((3.0/2.0)*dot_product(d_dev_stress(1:NSTR)
        *	   ,matmul(Matrix_P,d_dev_stress(1:NSTR))))
        p=sum(stress(1:Ndirect))/3.0
        e_stress(1:NSTR)=stress(1:NSTR)
        *	             -sum(stress(1:Ndirect))/3.0*delta(1:NSTR) !deviatori# stresses
        q = sqrt(1.5*dot_product(e_stress(1:NSTR)
        *	   ,matmul(Matrix_P,e_stress(1:NSTR))))	
        dEb=theta*(p*d_p/K+q*d_q/(3.0*G))/(1.0-theta*B)**2.0
        *   +theta**2.0*(p**2.0/K+q**2.0/(3.0*G))
        *   *dB/(1.0-theta*B)
        Eb=Ebtri+dEb
    #   deps_v=eps_v-eps_v0
    #   de=-deps_v*(1.0+e)
    #      dn=-deps_v/(1.0-n0)
    #   n=n+dn
    #      e=e+de
    #   nchange=nchange-(1.0+n)*deps_v/(1.0+n0)
# 
      if (i==1.or.mod(i,1.0)==0.or.i==t):
        write(6,*) i,B,p,q,eps_pv,eps_ev,eps_v,eps_es, eps_s, eps_ps	     
    #   write(7,*) e,p,q 
    #   write(8,*) n,p,q
    #   write(9,*) nchange,p,q
    #   write(11,*) i,strain(1),stress(1) 
    #   write(12,*) i,eps_v, p
    #   write(13,*) i,B,ytri
    #   write(14,*) i,eps_s,q
    #   write(16,*) i, strain(1), stress_tri(1)
    #   write(17,*) B,eps_pv,eps_ev
    #     write(18,*) strain(4),stress(4)
    #   write(19,*) dev_stress_tri(1),dev_stress_tri(2),dev_stress_tri(3)
    #     write(20,*) dev_stress_tri(4),dev_stress_tri(5),dev_stress_tri(6)
    #   write(21,*) dqtri_dsigmatri
    #      write(22,*) Ebtri,EEbtri,ytri
    #   write(23,*) theta,p,q
    #   write(24,*) eps_s,q,P#   write(25,*) q,Pc,B

100   continue
      end

