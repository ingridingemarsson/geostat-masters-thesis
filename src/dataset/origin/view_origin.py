import xarray as xr

dataset = xr.open_dataset('linkfileYYYY-MM/GPMGOES-oS2017-12-29T06:47:53-c[8,9,10,11,13,14,15,16]-p256/GPMGOES-oS2017-12-29T06:47:53-c[8,9,10,11,13,14,15,16]-p256-b0.nc')
print(dataset['gpm_precipitation'])
channels = list(range(8,17))
channels.remove(12)
for c in channels:
	print(dataset['C'+str(c).zfill(2)])
print(dataset.attrs['filenames_goes'])
print(dataset.attrs['goes_time_in'])
print(dataset.attrs['goes_time_out'])
print(dataset)
