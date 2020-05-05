      SUBROUTINE interpolate(arr, t_sim, t_slice, arr_x, arr_y, 
     &arr_z, c, res)
            integer :: t_sim, t_slice, arr_x, arr_y, arr_z
            double precision, intent(in):: arr(t_sim, arr_x, 
     &arr_y, arr_z)
      double precision, DIMENSION(3)::c
            double precision::res, x, y, z
            integer, DIMENSION(4)::small_c, s
            logical d, e, f

            res = 0
            s = shape(arr)
            x = c(1)
            y = c(2)
            z = c(3)
            d = x < 0 .or. x > s(2)
            e = y < 0 .or. y > s(3)
            f = z < 0 .or. z > s(4)
            if(d .or. e .or. f) then
                  return 
            end if
            small_c(1) = int(x)
            small_c(2) = int(y)
            small_c(3) = int(z)

            c000 = arr(t_slice, small_c(1), small_c(2), small_c(3))
            c001 = arr(t_slice,small_c(1), small_c(2), small_c(3)+1)
            c010 = arr(t_slice,small_c(1), small_c(2)+1, small_c(3))
            c011 = arr(t_slice,small_c(1), small_c(2)+1, small_c(3)+1)
            c100 = arr(t_slice,small_c(1)+1, small_c(2), small_c(3))
            c101 = arr(t_slice,small_c(1)+1, small_c(2), small_c(3)+1)
            c110 = arr(t_slice,small_c(1)+1, small_c(2)+1, small_c(3))
            c111 = arr(t_slice,small_c(1)+1, small_c(2)+1, small_c(3)+1)


            xd = x-small_c(1)
            yd = y-small_c(2)
            zd = z-small_c(3)

            c00 = c000*(1-xd) + c100*xd
            c01 = c001*(1-xd) + c101*xd
            c10 = c010*(1-xd) + c110*xd
            c11 = c001*(1-xd) + c111*xd

            c0 = c00*(1-yd) + c10*yd
            c1 = c01*(1-yd) + c11*yd
            res = c0*(1-zd) + c1*zd
            print*, xd, yd, zd
            ! write(*,*) c, 'result: ', res

      end
      

      SUBROUTINE write_arr(arr, t_sim, arr_x, arr_y, 
     &            arr_z, region)
            character (len = 80) :: name, form, access  
            integer :: t_sim, arr_x, arr_y, arr_z, recl
            double precision:: region(4)
            integer :: region_int(4)
            integer :: count_i, count_j
            double precision scale
            double precision, intent(in):: arr(t_sim, arr_x, 
     &            arr_y, arr_z)
            double precision, dimension (:,:), allocatable :: dout   
            double precision :: coord_conv(3)
            double precision slice

            slice = 75.
            scale = 4.

            do i = 1,4
                  region_int(i) = int(region(i))
            end do

            recl = (region_int(2) - region_int(1))*
     &      (region_int(4) - region_int(3))
                  form = 'unformatted'
                  access = 'direct'
                  name = 'C:\peter_abaqus\Summer-Research-Project
     &\test_fortran\v_500_s_151_single.mpout' 

            allocate (dout(region_int(2) - region_int(1), 
     &      region_int(4) - region_int(3)) )  
                  print*, 'region is: ' ,region_int
                  count_i = 1
                  count_j = 1
                  do i = region_int(1), (region_int(2)-1)
                        count_j = 1
                        do j = region_int(3), region_int(4)-1
                              coord_conv = (/i/scale, j/scale, slice/)
                              call interpolate(arr, t_sim, t_sim - 1,
     &                         arr_x, arr_y, arr_z, coord_conv, 
     &                         dout(count_i, count_j))
                              count_j = count_j + 1
                        end do      
                  count_i= count_i+1
                  end do
            print*, 'record is: ', recl
            open (unit=2,  file=name, form=form,  access=access,
     &      recl=recl*2,Status='REPLACE')
c     open (unit=1,  file='C:\peter_abaqus\Summer-Research-Project\data\v_500_s_15.mpout', 
c    &form='unformatted',  access='direct', recl=10*m*n*l*2)
            print*, 'opened'
            write (2, rec=1) dout
            close(2)
            print *, 'file is ready'
            deallocate (dout)  
            return
      end



      SUBROUTINE cov_coord(coords, coord_conv)
            double precision :: coords
            double precision resolution, size_cell
            double precision ::coord_conv
            integer scale

            scale = 4

            resolution = 15.
            size_cell = 5.*scale
            ! print*, 'inside cov coord', coords
            coord_conv = (coords + size_cell)* resolution
            ! print*, 'inside cov coord', coord_conv
            return 
      end
      
      SUBROUTINE DFLUX(FLUX,COORDS,SOL,JSTEP,JINC,TIME,NOEL,
     &NPT,JLTYP,TEMP,PRESS,SNAME)         
      ! INCLUDE 'ABA_PARAM.INC' 
      DIMENSION FLUX(2),TIME(2)
      double precision :: COORDS(3)
      double precision coord_conv(3)
      integer t_sim, t_slice, m,n,l
      parameter (t_sim = 16, t_slice = 15, m=150, n=150, l =150)
      double precision, save :: data_out(t_sim, m, n, l)
      integer, save:: r_file = 1
      integer, save:: counter = 1
      real t, total_t, meep_abq_conv_rate
      character (len = 80) :: name, form, access  
      integer recl
      double precision :: region(4), cov_region(4)
      integer scale


      total_t = 1.
      meep_abq_conv_rate = 10
      recl = t_sim*m*n*l*2

      form = 'unformatted'
      access = 'direct'
      name = 'C:\\peter_abaqus\\Summer-Research-Project
     &\\data\\v_500_s_15.clean' 
            counter = counter  + 1

            if (r_file == 1) then
                  open (unit=1,  file=name, form=form,  
     &            access=access, recl=recl)
                  read (1, rec=1) data_out
                  close(1)
                  print *, 'file is ready'
                  r_file = r_file + 1
            end if
      
c use code to define DDSDDE, STRESS, STATEV, SSE, SPD, SCD, and if necessary, RPL, DDSDDT, DRPLDE, DRPLDT, PNEWDT

            do i = 1, 3
                  call cov_coord(COORDS(i), coord_conv(i))
            end do

            scale = 4
            region(1) = -3.*scale
            region(2) = 3.*scale
            region(3) = -3.*scale
            region(4) = 3.*scale

            do i = 1, 4
                  call cov_coord(region(i), cov_region(i))
            end do

            print *, 'region is ', cov_region
            
            call interpolate(data_out, t_sim, t_slice,
     &m, n, l, coord_conv, flux(1))
            flux(1) = flux(1)*10e3
            print*, flux(1)


            call write_arr(data_out, t_sim,
     &m, n, l, cov_region)


            FLUX(2) = 0
      

            RETURN
            END

c # shutil.copyfile('C:/temp/dflux.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/dflux.inp')
c # shutil.copyfile('C:/temp/umat_test.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/umat_test.inp')


      PROGRAM SUBDEM 
      REAL A,B,C,SUM,SUMSQ 
      DIMENSION  FLUX(2)
      double precision :: COORDS(3)

      COORDS(1) = 0.
      COORDS(2) = 1.
      COORDS(3) = 2.
      CALL DFLUX(flux, COORDS)
      END
