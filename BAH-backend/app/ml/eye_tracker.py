import numpy as np
import cv2
import math

def detect_eye(tir_channel: np.ndarray, prev_x=None, prev_y=None, vx=0, vy=0):
    """
    Detects the cyclone eye in a TIR (Kelvin) image.
    Uses fallback hierarchy:
    1. Warmest local max (confidence 0.9)
    2. Min-variance 31x31 patch (confidence 0.6)
    3. Motion extrapolation (confidence 0.3)
    """
    h, w = tir_channel.shape
    
    # Pre-mask NaN values
    tir_clean = np.nan_to_num(tir_channel, nan=np.nanmin(tir_channel))
    
    # Look for the storm center (coldest region)
    blurred = cv2.GaussianBlur(tir_clean, (31, 31), 0)
    storm_y, storm_x = np.unravel_index(np.argmin(blurred), blurred.shape)
    
    # 1. Warmest local maximum inside the storm region
    radius = 40
    y_start, y_end = max(0, storm_y - radius), min(h, storm_y + radius)
    x_start, x_end = max(0, storm_x - radius), min(w, storm_x + radius)
    
    storm_patch = tir_clean[y_start:y_end, x_start:x_end]
    if storm_patch.size > 0:
        local_y, local_x = np.unravel_index(np.argmax(storm_patch), storm_patch.shape)
        eye_y, eye_x = y_start + local_y, x_start + local_x
        
        # Check if it's truly a "warm" eye (e.g. warmer than the blurred CDO by some threshold)
        if tir_clean[eye_y, eye_x] > blurred[storm_y, storm_x] + 5.0:
            return eye_x, eye_y, 0.9
            
    # 2. Min-variance 31x31 patch
    # If eye is not clearly warm, maybe it's just a calm patch
    best_var = float('inf')
    best_x, best_y = storm_x, storm_y
    patch_r = 15
    for y in range(max(patch_r, storm_y - radius), min(h - patch_r, storm_y + radius)):
        for x in range(max(patch_r, storm_x - radius), min(w - patch_r, storm_x + radius)):
            patch = tir_clean[y-patch_r:y+patch_r+1, x-patch_r:x+patch_r+1]
            var = np.var(patch)
            if var < best_var:
                best_var = var
                best_x, best_y = x, y
                
    if best_var < 50.0:  # arbitrary threshold for low variance
        return best_x, best_y, 0.6
        
    # 3. Motion extrapolation
    if prev_x is not None and prev_y is not None:
        extrap_x = int(np.clip(prev_x + vx, 0, w - 1))
        extrap_y = int(np.clip(prev_y + vy, 0, h - 1))
        return extrap_x, extrap_y, 0.3
        
    # Ultimate fallback: just return the coldest storm center
    return int(storm_x), int(storm_y), 0.1

def calculate_bearing(lat0, lon0, lat1, lon1):
    """Calculate bearing from (lat0, lon0) to (lat1, lon1)"""
    lat1_rad = math.radians(lat1)
    lat0_rad = math.radians(lat0)
    diff_long = math.radians(lon1 - lon0)
    x = math.sin(diff_long) * math.cos(lat1_rad)
    y = math.cos(lat0_rad) * math.sin(lat1_rad) - (math.sin(lat0_rad) * math.cos(lat1_rad) * math.cos(diff_long))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def to_compass(degrees):
    val = int((degrees / 22.5) + .5)
    arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return arr[(val % 16)]

def calculate_motion(eye0, eye1, gap_hours, km_per_pixel=2.0, latlon0=None, latlon1=None):
    x0, y0 = eye0
    x1, y1 = eye1
    
    if latlon0 is not None and latlon1 is not None:
        lat0, lon0 = latlon0
        lat1, lon1 = latlon1
        
        distance_km = haversine_distance(lat0, lon0, lat1, lon1)
        speed_kmh = distance_km / max(gap_hours, 0.01)
        
        compass_deg = calculate_bearing(lat0, lon0, lat1, lon1)
        compass = to_compass(compass_deg)
        
        direction_deg = (90 - compass_deg) % 360
        
        vx_kmh = speed_kmh * math.sin(math.radians(compass_deg))
        vy_kmh = speed_kmh * math.cos(math.radians(compass_deg))
    else:
        # Pixel displacement fallback
        dx = x1 - x0
        dy = y1 - y0
        
        distance_km = math.sqrt(dx**2 + dy**2) * km_per_pixel
        speed_kmh = distance_km / max(gap_hours, 0.01)
        
        # Direction (flip dy for compass since image y increases downward)
        direction_deg = math.degrees(math.atan2(-dy, dx))
        compass_deg = (90 - direction_deg) % 360  # convert math angle to compass bearing
        compass = to_compass(compass_deg)
        
        vx_kmh = speed_kmh * math.sin(math.radians(compass_deg))
        vy_kmh = speed_kmh * math.cos(math.radians(compass_deg))
    
    return {
        "speed_kmh": round(speed_kmh, 1),
        "vx_kmh": round(vx_kmh, 1),
        "vy_kmh": round(vy_kmh, 1),
        "distance_km": round(distance_km, 1),
        "direction_deg": round(direction_deg, 1),
        "compass": compass
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

import xarray as xr
def get_lat_lon_from_pixel(nc_path, x_pixel, y_pixel):
    """
    Safely converts a pixel coordinate (x_pixel, y_pixel) in a GOES NetCDF 
    to geographical (latitude, longitude) coordinates.
    Returns (lat, lon) or None if metadata is missing.
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        if 'x' not in ds or 'y' not in ds or 'goes_imager_projection' not in ds:
            ds.close()
            return None
            
        x_rad = ds.x.values[x_pixel]
        y_rad = ds.y.values[y_pixel]
        
        proj_info = ds.goes_imager_projection
        h = proj_info.perspective_point_height
        lon_0 = proj_info.longitude_of_projection_origin
        semi_major = proj_info.semi_major_axis
        semi_minor = proj_info.semi_minor_axis
        
        from pyproj import Proj
        p = Proj(proj='geos', h=h, lon_0=lon_0, sweep='x', a=semi_major, b=semi_minor)
        
        x_meters = x_rad * h
        y_meters = y_rad * h
        
        lon, lat = p(x_meters, y_meters, inverse=True)
        ds.close()
        
        if np.isnan(lon) or np.isnan(lat) or lon > 180 or lon < -180:
            return None
            
        return float(lat), float(lon)
    except Exception as e:
        print(f"Failed to get lat/lon: {e}")
        return None
