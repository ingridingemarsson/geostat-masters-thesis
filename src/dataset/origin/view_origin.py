import xarray as xr

dataset = xr.open_dataset('linkfile2017-12/GPMGOES-oS2017-12-31T17:26:29-c[8,9,10,11,13,14,15,16]-p256/GPMGOES-oS2017-12-31T17:26:29-c[8,9,10,11,13,14,15,16]-p256-b1.nc')
#print(dataset['gpm_precipitation'])
#channels = list(range(8,17))
#channels.remove(12)
#for c in channels:
#	print(dataset['C'+str(c).zfill(2)])
#print(dataset.attrs['filenames_goes'])
#print(dataset.attrs['goes_time_in'])
#print(dataset.attrs['goes_time_out'])

print(dataset)
print(dataset.attrs['filenames_goes'])

dataset = xr.open_dataset('linkfile2017-12/GPMGOES-oS2017-12-31T17:26:29-c[8,9,10,11,13,14,15,16]-p256/GPMGOES-oS2017-12-31T17:26:29-c[8,9,10,11,13,14,15,16]-p256-b0.nc')
print(dataset)
print(dataset.attrs['filenames_goes'])


dataset = xr.open_dataset('linkfile2017-12/GPMGOES-oS2017-12-31T17:26:29-c[8,9,10,11,13,14,15,16]-p256/GPMGOES-oS2017-12-31T17:26:29-c[8,9,10,11,13,14,15,16]-p256-b-1.nc')
print(dataset)
print(dataset.attrs['filenames_goes'])

