! timeseries_reader.f90
subroutine load_timeseries

  use module_global, only: filetimeseries, ts_Q, ts_W, ts_D, ts_E_rate, &
                           ts_physical_ds, n_timeseries, dirIN
  implicit none

  integer :: lun, io_stat, i
  character(len=200) :: line ! Buffer to read lines
  character(len=200) :: fullpath ! Full path to timeseries file
  character(len=1) :: comma ! Delimiter
  real*8 :: month_read ! Temporary variable for month column

  lun = 50 ! Assign a free logical unit number
  comma = ','

  ! Construct full path using dirIN prefix (same as load_river.f90)
  fullpath = TRIM(dirIN) // TRIM(filetimeseries)

  open(unit=lun, file=TRIM(fullpath), status='old', action='read', iostat=io_stat)
  if (io_stat /= 0) then
    write(6,*) 'ERROR: Could not open timeseries file: ', TRIM(filetimeseries)
    stop 1
  end if

  ! Read header line and discard
  read(lun, '(A)', iostat=io_stat) line
  if (io_stat /= 0) then
     write(6,*) 'ERROR: Could not read header from timeseries file: ', TRIM(filetimeseries)
     close(lun)
     stop 1
  end if

  ! Count the number of data lines first to allocate arrays
  n_timeseries = 0
  do
    read(lun, '(A)', iostat=io_stat) line
    if (io_stat /= 0) exit ! Exit loop on EOF or error
    n_timeseries = n_timeseries + 1
  end do

  if (n_timeseries == 0) then
      write(6,*) 'ERROR: No data found in timeseries file: ', TRIM(filetimeseries)
      close(lun)
      stop 1
  end if

  ! Allocate arrays
  allocate ( ts_Q(n_timeseries), ts_W(n_timeseries), ts_D(n_timeseries), &
             ts_E_rate(n_timeseries), ts_physical_ds(n_timeseries), stat=io_stat )
  if (io_stat /= 0) then
    write(6,*) 'ERROR: Could not allocate memory for timeseries data.'
    close(lun)
    stop 1
  end if

  ! Rewind the file and re-read data into arrays
  rewind(lun)
  read(lun, '(A)') line ! Read and discard header again

  do i = 1, n_timeseries
    ! Assuming format: month,Q,W,D,E_rate,physical_ds
    read(lun, *, iostat=io_stat) month_read, ts_Q(i), ts_W(i), ts_D(i), ts_E_rate(i), ts_physical_ds(i)
    ! Alternative stricter read with comma delimiter if needed:
    ! read(lun, '(F8.0, A1, F12.0, A1, F12.0, A1, F12.0, A1, E12.6, A1, F12.9)', iostat=io_stat) &
    !      month_read, comma, ts_Q(i), comma, ts_W(i), comma, ts_D(i), comma, ts_E_rate(i), comma, ts_physical_ds(i)

    if (io_stat /= 0) then
      write(6,*) 'ERROR: Reading data line ', i+1, ' from timeseries file.'
      close(lun)
      deallocate(ts_Q, ts_W, ts_D, ts_E_rate, ts_physical_ds)
      stop 1
    end if
  end do

  close(lun)

  write(6,*) 'Successfully read ', n_timeseries, ' monthly records from ', TRIM(filetimeseries)

contains

! Add internal functions if needed, e.g., for error handling

end subroutine load_timeseries 