import h5py
import numpy as np
import netCDF4 as nc

rootgrp = nc.Dataset("test.nc", "w", format="NETCDF4")
x = rootgrp.createDimension("x", 1)
y = rootgrp.createVariable("y","f4",("x",))
y[:] = 0

print('Created netCDF4 file test.nc')