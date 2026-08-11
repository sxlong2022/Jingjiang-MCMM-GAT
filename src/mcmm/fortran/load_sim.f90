!-----------------------------------------------------------------------
! LOAD FILE SIM: river and simulation parameters
!----------------------------------------------------------------------- 
      subroutine load_sim

      use module_global
      use module_zs
      implicit none

! open file
      filesim = ADJUSTL(filesim)
      filename = TRIM(dirIN)//TRIM(filesim)
      open(unit=1, file=filename, status='old')
      
! intro
      read(1,*)
      read(1,*) simname    ! name of the simulation

! morphodynamics
      read(1,*)
      read(1,*) beta0      ! aspect ratio
      read(1,*) theta0     ! Shields number
      read(1,*) ds0        ! grain roughness
      read(1,*) flagbed    ! bedset (1:from Rp, 2:from bed input)
      read(1,*) Rp         ! particle Reynolds Rp=(Delta g ds^3)^0.5/ni
      read(1,*) rpic0      ! transverse transport parameter (Talmon)
      read(1,*) jmodel     ! flag for flow field model (1:ZS, 2:IPS)
      read(1,*) Nz         ! number of points for the vertical integration
      read(1,*) Mdat       ! order of Fourier expansion

! geomorphology
      read(1,*)
      read(1,*) Ef         ! erodibility coefficient of the floodplain
      read(1,*) Eb         ! erodibility coefficient of the point bars
      read(1,*) Eo         ! erodibility coefficient of the oxbow lakes
      read(1,*) flag_ox    ! flag for existing floodplain structure (0:no, 1:yes)

! river configuration
      read(1,*)
      read(1,*) N0         ! initial number of points
      read(1,*) flagxy0    ! initial path configuration (1:random, 2:given)
      read(1,*) filexy     ! name of geometry file
      read(1,*) deltas0    ! distance between axis points
      read(1,*) dsmin      ! min value of grid size (times deltas0)
      read(1,*) dsmax      ! max value of grid size (times deltas0)
      read(1,*) Nrand      ! number of point interested by a perturbation
      read(1,*) stdv       ! standard deviation of initial perturbation
      read(1,*) tollc      ! minimum threshold before neck-cutoff
      read(1,*) jre        ! removed points before and after a cutoff
      read(1,*) jnco       ! minimum threshold points for neck cutoff
      read(1,*) flag_cutoff ! flag for cutoff detection (0:disabled, 1:enabled)
      read(1,*) ksavgol    ! Savitzky-Golay flag
      
! time step and printing
      read(1,*)
      read(1,*) flag_time  ! final time assignment
      read(1,*) TTs        ! simulation time if flag_time = 1
      read(1,*) kTTfco     ! coefficient for first cutoff time
      read(1,*) nend       ! item number (iterations, cutoffs, printed confs)
      read(1,*) tt0        ! starting time of simulation
      read(1,*) flag_dt    ! flag for time marching
      read(1,*) dt0        ! fixed time step
      read(1,*) cstab      ! coefficient for time marching
      read(1,*) ivideo     ! number of iterations between two video prints
      read(1,*) ifile      ! number of iterations between two files prints
      
! valley boundaries
      read(1,*)
      read(1,*) jbound         ! transition shape
      read(1,*) Ebound         ! erodibility coefficient of valley boundaries
      read(1,*) Lhalfvalley    ! transverse half-width of floodplain
      read(1,*) Ltransvalley   ! thickness of the transition layer

! NEW: timeseries parameters (optional section)
! Read additional lines if they exist
      filetimeseries = ''
      slope = 0.0d0
      delta_sg = 1.65d0
      physical_ds = 0.0d0
      E_scale = 1.0d0
      factor_Eb = 1.0d0
      factor_Eo = 1.0d0
      kinematic_viscosity = 1.0d-6
      base_Ef = 0.0d0
      
      read(1,*,end=100,err=100)
      read(1,*,end=100,err=100) filetimeseries
      read(1,*,end=100,err=100) slope
      read(1,*,end=100,err=100) delta_sg
      read(1,*,end=100,err=100) physical_ds
      read(1,*,end=100,err=100) E_scale
      read(1,*,end=100,err=100) factor_Eb
      read(1,*,end=100,err=100) factor_Eo
      read(1,*,end=100,err=100) kinematic_viscosity

100   continue
      close(1)

! Set initial values
      beta = beta0
      theta = theta0
      ds = ds0
      rpic = rpic0

! Check timeseries parameters if file specified
      if (LEN_TRIM(filetimeseries) > 0) then
         if (slope <= 0.0d0) then
            write(6,*) 'ERROR: slope must be positive for timeseries'
            stop
         end if
         if (physical_ds <= 0.0d0) then
            write(6,*) 'ERROR: physical_ds must be positive for timeseries'
            stop
         end if
      end if

! print summary on screen
      call dashline(6)
      write(6,*) 'INTRO'
      write(6,*) 'simulation name : ', simname
      select case(flagxy0)
      case(1)
        write(6,*) 'straight river with slight random perturbation'
      case(2)
        write(6,*) 'coordinate file : ', filexy
      case default
        stop 'ERROR! Wrong flag for initial river configuration'
      end select
      
! end of the subroutine
      return
      end
