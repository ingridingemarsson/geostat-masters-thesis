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
    C08                (y, x) float32 234.3 234.7 234.5 ... 239.5 239.8 239.9
    C13                (y, x) float32 281.8 284.8 285.5 ... 288.3 288.6 288.7
    gpm_precipitation  (y, x) float32 nan nan nan nan nan ... nan nan nan nan
Attributes:
    ind_extent:      [1816 2185 2072 1929]
    area_extent:     [ 1845699.92226294 -3324664.68081892  2871756.77372725 -...
    shape:           [256 256]
    gpm_time_in:     2017-12-29 08:03:23.862000
    gpm_time_out:    2017-12-29 08:07:07.161999
    goes_time_in:    2017-12-29 08:00:45.500000
    goes_time_out:   2017-12-29 08:11:22.200000
    filename_gpm:    2B.GPM.DPRGMI.2HCSHv4-1.20171229-S064753-E082028.021790....
    filenames_goes:  ['OR_ABI-L1b-RadF-M3C08_G16_s20173630800455_e20173630811...
```
### 

<p float="left">
<img src="plots/C08af.png" width="400">
<img src="plots/C13af.png" width="400">
<img src="plots/gpmaf.png" width="400">
</p>


