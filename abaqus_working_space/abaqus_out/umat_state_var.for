      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, RPL,DDSDDT,DRPLDE,DRPLDT, STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED, 
     & CMNAME, NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT, CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)

c23456789 (This demonstrates column position!)         
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV), DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS), STRAN(NTENS),
     & DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1), PROPS(NPROPS),COORDS(3)
     & ,DROT(3,3),DFGRD0(3,3),DFGRD1(3,3), JSTEP(4)


      parameter (one = 1.0d0, two=2.0d0)

      REAL x, y, sigma
      parameter (sigma = 10e9)
      
      real z
      integer seed(1)
      
      real :: pi = 3.1415926535485009
c23456789 (This demonstrates column position!)    
      
      real coord_conv(3)
      integer m,n,l
      parameter (m=150, n=150, l =150)
      double precision, save :: data_out(33, m, n, l)
      integer, save:: r_file = 1
      integer, save:: counter = 1
      real t
      counter = counter  + 1

      if (r_file == 1) then
            open (unit=1,  file='C:\peter_abaqus\Summer-Research-Project\meep\meep_out\prism_dis_0.2.bin', 
     &form='unformatted',  access='direct', recl=33*m*n*l*2)
            read (1, rec=1) data_out
            close(1)
            print *, 'file is read'
            r_file = r_file + 1
      end if

      seed(1) =  10
      CALL RANDOM_SEED(put=seed)
      
      CALL RANDOM_NUMBER(x)
      call RANDOM_NUMBER(y)
      z = sqrt(-2*log(x))*cos(2*pi*y)

      
c use code to define DDSDDE, STRESS, STATEV, SSE, SPD, SCD, and if necessary, RPL, DDSDDT, DRPLDE, DRPLDT, PNEWDT

      
      coord_conv(1) = ceiling((coords(1)+0.5)*m)
      coord_conv(2) = ceiling((coords(2)+0.5)*n)
      coord_conv(3) = ceiling((coords(3)+0.5)*l)

      STATEV(1) = data_out(15, int(coord_conv(1)), int(coord_conv(2)), int(coord_conv(3)))
      ! print*, int(coord_conv(1)), int(coord_conv(2)), int(coord_conv(3))
      print*, statev(1)

      E = props(1) + z*sigma
      ANU = props(2)
      ALAMDA = E*ANU/(1+ANU)/(ONE - TWO*ANU)
      AMU = E/(ONE+ANU)/2
      DO i = 1,NTENS
         DO j = 1,NTENS
            ddsdde(i,j) = 0.0d0
         end do
      end do
      ddsdde(1, 1) = (ALAMDA + two*amu)
      ddsdde(2, 2) = (ALAMDA + two*amu)
      ddsdde(3, 3) = (ALAMDA + two*amu)
      ddsdde(4, 4) = amu
      ddsdde(5, 5) = amu
      ddsdde(6, 6) = amu
      ddsdde(1, 2) = ALAMDA
      ddsdde(1, 3) = ALAMDA
      ddsdde(2, 3) = ALAMDA
      ddsdde(2, 1) = ALAMDA
      ddsdde(3, 1) = ALAMDA
      ddsdde(3, 2) = ALAMDA

      do i = 1, NTENS
         do j = 1, NTENS
            stress(i) = stress(i) + ddsdde(i,j) * DSTRAN(j)
         end do
      end do

    
c      ! t = int(ceiling(time(1)*10))

!      print*, int(coord_conv(1)), int(coord_conv(2)), int(coord_conv(3))
      ! STATEV(2) = 2.
      ! STATEV(3) = 3.
      ! STATEV(4) = 4.
      return 
      end subroutine

      