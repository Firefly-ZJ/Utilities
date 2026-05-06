### Convert degrees to area in km²

import numpy as np
import warnings

Radius = 6371.0  # Earth radius (km)
Pi = np.pi

def _pixel_area(lat:float, lat_res:float, lon_res:float=None) -> float:
    """Calculate pixel area in km².
    `Area = R**2 * ΔLon * Δsin(Lat)`

    Args:
        lat (float): pixel latitude (left bottom corner)
        lat_res (float): latitude resolution
        lon_res (float, optional): longitude resolution (defaults to lat_res)
    """
    if lon_res is None:  lon_res = lat_res
    alpha, beta = np.deg2rad(lat), np.deg2rad(lat + lat_res)
    delta = np.deg2rad(lon_res)
    area = Radius**2 * delta * (np.sin(beta) - np.sin(alpha))
    return area

def get_weighted(data:np.ndarray, lat_range:tuple, lon_range:tuple) -> np.ndarray:
    """Calculate the area-weighted (in km²) array of given data.

    Args:
        data (ndarray): 2D data array of [lat, lon]
        lat_range (tuple): 纬度范围 latitude range (min, max)
        lon_range (tuple): 经度范围 longitude range (min, max)
    
    Returns:
        weighted (ndarray): 2D area-weighted data array of [lat, lon]
    """
    if not data.ndim == 2:
        raise ValueError("Input data must be 2D array of [lat, lon]")
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    row, col = data.shape

    lats = np.linspace(lat_min, lat_max, row+1)
    lat_res = (lat_max - lat_min) / row  # pixel resolution
    lon_res = (lon_max - lon_min) / col

    func = lambda lat: _pixel_area(lat, lat_res, lon_res)
    areas = np.array(list(map(func, lats[:-1])))
    weighted = data * areas[:, np.newaxis]
    return weighted

def get_sumup(data:np.ndarray, lat_range:tuple, lon_range:tuple, factor:float=1.0) -> float:
    """Calculate the area-weighted (in km²) sum of given data.

    Args:
        data (ndarray): 2D data array of [lat, lon]
        lat_range (tuple): 纬度范围 latitude range (min, max)
        lon_range (tuple): 经度范围 longitude range (min, max)
        factor (float, optional): scale factor
    
    Returns:
        sumup (float): area-weighted sum of data array, multiplied by factor
    
    Note: Sum up longitude dimension first.
    """
    if not data.ndim == 2:
        raise ValueError("Input data must be 2D array of [lat, lon]")
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    row, col = data.shape

    lats = np.linspace(lat_min, lat_max, row+1)
    lat_res = (lat_max - lat_min) / row  # pixel resolution
    lon_res = (lon_max - lon_min) / col

    func = lambda lat: _pixel_area(lat, lat_res, lon_res)
    areas = np.array(list(map(func, lats[:-1])))
    with warnings.catch_warnings(action="ignore"):
        data_1d = np.nansum(data, axis=1)  # sum up longitude dimension first
        weighted = data_1d * areas
        sumup = np.sum(weighted)  # sum up latitude dimension
    return sumup * factor

if __name__ == "__main__":
    grids = np.ones((180, 360), dtype=np.float64)
    area = get_weighted(grids, (-90, 90), (-180, 180))
    print(area.shape)
    
    print(area.sum() / 10**8)  # 5.101 * 10^8 km²
    print(get_sumup(grids, (-90, 90), (-180, 180)) / 10**8)
