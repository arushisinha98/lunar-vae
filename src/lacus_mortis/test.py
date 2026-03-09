import numpy as np
import os
from glob import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import warnings
import re

# Parameters (should match preprocess.py)
GP_NOISE_K = 10.0
GP_LS_MAX_HR = 6.0
GP_SAMPLE_DT = 0.2
GP_N_SAMPLES = 120
T_GRID = np.arange(GP_N_SAMPLES) * GP_SAMPLE_DT

def gp_interpolate(ltime, temp, noise_k=GP_NOISE_K, ls_max=GP_LS_MAX_HR, t_query=T_GRID):
	X = ltime.reshape(-1, 1)
	y = temp.astype(np.float64)
	kernel = (
		Matern(length_scale=3.0, length_scale_bounds=(0.1, ls_max), nu=1.5)
		+ WhiteKernel(noise_level=noise_k**2, noise_level_bounds="fixed")
	)
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			gp.fit(X, y)
		pred = gp.predict(t_query.reshape(-1, 1))
		return pred.astype(np.float32)
	except Exception as exc:
		print(f"GP failed: {exc}")
		return None

# Gather all .xyz files
xyz_files = sorted(glob("data/lacus_mortis/raw_xyz/lacus_mortis-tb-*.xyz"))
print(f"Found {len(xyz_files)} files.")

# Find the first (lon, lat) in the first file
first_coord = None
for f in xyz_files:
    try:
        arr = np.loadtxt(f, comments=["#", "%"], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > 0:
            first_coord = tuple(np.round(arr[0, :2], 4))
            break
    except Exception as e:
        continue
if first_coord is None:
    print("No valid data found in any file.")
    exit(1)

print(f"Using first (lon, lat): {first_coord}")

# Collect (local_time, temp) for this coord only
ltime_list = []
temp_list = []
for f in xyz_files:
	try:
		arr = np.loadtxt(f, comments=["#", "%"], dtype=np.float32)
		if arr.ndim == 1:
			arr = arr[None, :]
		# Infer local_time from filename index (as in preprocess.py)
		m = re.search(r"lacus_mortis-tb-(\d{3})\.xyz", f)
		if m:
			time_idx = int(m.group(1))
			local_time = (time_idx - 1) * 0.1
		else:
			local_time = 0.0
		for row in arr:
			lon, lat, temp = row
			if tuple(np.round([lon, lat], 4)) == first_coord:
				ltime_list.append(local_time)
				temp_list.append(temp)
	except Exception as e:
		continue

ltime = np.array(ltime_list)
temp = np.array(temp_list)

if len(ltime) == 0:
	print("No data found for the selected (lon, lat).")
	exit(1)

print("Local times:", ltime)
print("Temperatures:", temp)

# Sort by local_time
order = np.argsort(ltime)
ltime = ltime[order]
temp = temp[order]

# Remove NaNs
valid = ~np.isnan(temp)
ltime = ltime[valid]
temp = temp[valid]

# GP interpolation
profile_interp = gp_interpolate(ltime, temp)
print("\nGP interpolated profile:")
print(profile_interp)