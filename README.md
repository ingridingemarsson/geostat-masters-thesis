# Master's Thesis in Space, earth and environment
## Spring 2021

Downloading data for rain retrievals over Brazil.


### Workflow preprocessing
- Read link from linkfile and convert to start, end date
- Download gpm product for start, end date
- Transform gpm data into goes data coordinates
- Divide interesting data into NxN pixel boxes along swath
- Extract in, out time for gpm passing box region
- Download goes file matching box time
- Combine gpm and goes box data into Dataset



### Dataset storage format
```python
<xarray.Dataset>
Dimensions:            (x: 256, y: 256)
Dimensions without coordinates: x, y
Data variables:
    gpm_precipitation  (y, x) float32 nan nan nan nan nan ... nan nan nan nan
    C08                (y, x) float32 234.1 234.2 234.6 ... 231.5 230.5 228.6
    C13                (y, x) float32 292.3 292.9 293.7 ... 280.0 276.9 272.4
Attributes:
    ind_extent:      [1570 1449 1826 1193]
    area_extent:     [ 859723.41655896 -374751.23285903 1885780.26802327  651...
    shape:           [256 256]
    gpm_time_in:     2018-10-31T14:25:30.382000128
    gpm_time_out:    2018-10-31T14:28:24.681999872
    filename_gpm:    GPM/2B.GPM.DPRGMI.2HCSHv4-1.20181031-S131753-E145026.026...
    goes_time_in:    2018-10-31 14:15:37
    goes_time_out:   2018-10-31 14:26:13.700000
    filenames_goes:  ['GOES-16/linkfile464113/OR_ABI-L1b-RadF-M3C08_G16_s2018...
```
### 

<p float="left">
<img src="plots/C08af.png" width="400">
<img src="plots/C13af.png" width="400">
<img src="plots/gpmaf.png" width="400">
</p>


