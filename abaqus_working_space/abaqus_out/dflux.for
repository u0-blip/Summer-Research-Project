      SUBROUTINE DFLUX(FLUX,SOL,JSTEP,JINC,TIME,NOEL,NPT,COORDS,JLTYP, TEMP,PRESS,SNAME)


      INCLUDE 'ABA_PARAM.INC'

      DIMENSION COORDS(3),FLUX(2),TIME(2)
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
      
c use code to define DDSDDE, STRESS, STATEV, SSE, SPD, SCD, and if necessary, RPL, DDSDDT, DRPLDE, DRPLDT, PNEWDT

      
      coord_conv(1) = ceiling((coords(1)+0.5)*m)
      coord_conv(2) = ceiling((coords(2)+0.5)*n)
      coord_conv(3) = ceiling((coords(3)+0.5)*l)

      flux(1) = data_out(15, int(coord_conv(1)), int(coord_conv(2)), int(coord_conv(3)))
      ! print*, int(coord_conv(1)), int(coord_conv(2)), int(coord_conv(3))
      print*, flux(1)
       ! t = int(ceiling(time(1)*10))

c      ! if (mod(counter, 100) == 0) then
c      !       print *, flux(1)
c      ! end if

c      ! flux(1) = 10e-03
      FLUX(2) = 0
 

      RETURN
      END

c # shutil.copyfile('C:/temp/dflux.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/dflux.inp')
c # shutil.copyfile('C:/temp/umat_test.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/umat_test.inp')