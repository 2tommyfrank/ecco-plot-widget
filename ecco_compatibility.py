import pandas as pd
import xarray as xr
import datetime as dt
import requests
import os
from netrc import netrc
from urllib import request
from http.cookiejar import CookieJar
from platform import system

downloads = os.path.expanduser('~/Downloads')
_netrc = os.path.join(os.path.expanduser('~'), '_netrc' if system()=='Windows' else '.netrc')

# Login to EarthData from netrc file
try:
    username, _, password = netrc(file=_netrc).authenticators('urs.earthdata.nasa.gov')
    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, 'urs.earthdata.nasa.gov', username, password)
    auth = request.HTTPBasicAuthHandler(manager)
    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)
    with requests.get('https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc') as r:
        if r.status_code // 100 == 2:
            print('Login to EarthData successful')
        elif r.status_code == 401:
            print('Incorrect EarthData login; check netrc')
        else:
            print(r.text)
except FileNotFoundError:
    print('netrc file not found')

# Information to look up a variable in EarthData by name
all_variables = ['global_mean_barystatic_sea_level_anomaly', 'global_mean_sterodynamic_sea_level_anomaly',
                 'global_mean_sea_level_anomaly', 'Pa_global', 'xoamc', 'yoamc', 'zoamc', 'xoamp', 'yoamp', 'zoamp',
                 'mass', 'xcom', 'ycom', 'zcom', 'sboarea', 'xoamc_si', 'yoamc_si', 'zoamc_si', 'mass_si', 'xoamp_fw',
                 'yoamp_fw', 'zoamp_fw', 'mass_fw', 'xcom_fw', 'ycom_fw', 'zcom_fw', 'mass_gc', 'xoamp_dsl',
                 'yoamp_dsl', 'zoamp_dsl', 'CS', 'SN', 'rA', 'dxG', 'dyG', 'Depth', 'rAz', 'dxC', 'dyC', 'rAw', 'rAs',
                 'drC', 'drF', 'PHrefC', 'PHrefF', 'hFacC', 'hFacW', 'hFacS', 'maskC', 'maskW', 'maskS', 'DIFFKR',
                 'KAPGM', 'KAPREDI', 'SSH', 'SSHIBC', 'SSHNOIBC', 'ETAN', 'EXFatemp', 'EXFaqh', 'EXFuwind', 'EXFvwind',
                 'EXFwspee', 'EXFpress', 'EXFtaux', 'EXFtauy', 'oceTAUX', 'oceTAUY', 'EXFhl', 'EXFhs', 'EXFlwdn',
                 'EXFswdn', 'EXFqnet', 'oceQnet', 'SIatmQnt', 'TFLUX', 'EXFswnet', 'EXFlwnet', 'oceQsw', 'SIaaflux',
                 'EXFpreci', 'EXFevap', 'EXFroff', 'SIsnPrcp', 'EXFempmr', 'oceFWflx', 'SIatmFW', 'SFLUX', 'SIacSubl',
                 'SIrsSubl', 'SIfwThru', 'SIarea', 'SIheff', 'SIhsnow', 'sIceLoad', 'SIuice', 'SIvice', 'ADVxHEFF',
                 'ADVyHEFF', 'DFxEHEFF', 'DFyEHEFF', 'ADVxSNOW', 'ADVySNOW', 'DFxESNOW', 'DFyESNOW', 'oceSPflx',
                 'oceSPDep', 'MXLDEPTH', 'OBP', 'OBPGMAP', 'PHIBOT', 'UVEL', 'VVEL', 'WVEL', 'THETA', 'SALT',
                 'RHOAnoma', 'DRHODR', 'PHIHYD', 'PHIHYDcR', 'UVELMASS', 'VVELMASS', 'WVELMASS', 'Um_dPHdx', 'Vm_dPHdy',
                 'ADVx_TH', 'ADVy_TH', 'ADVr_TH', 'DFxE_TH', 'DFyE_TH', 'DFrE_TH', 'DFrI_TH', 'ADVx_SLT', 'ADVy_SLT',
                 'ADVr_SLT', 'DFxE_SLT', 'DFyE_SLT', 'DFrE_SLT', 'DFrI_SLT', 'oceSPtnd', 'UVELSTAR', 'VVELSTAR',
                 'WVELSTAR', 'GM_PsiX', 'GM_PsiY']
all_datasets = ['GMSL_TIME_SERIES', 'GMAP_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'GEOMETRY_LLC0090GRID',
                'OCEAN_3D_MIX_COEFFS_LLC0090GRID', 'SSH_LLC0090GRID', 'ATM_STATE_LLC0090GRID', 'STRESS_LLC0090GRID',
                'HEAT_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID', 'SEA_ICE_CONC_THICKNESS_LLC0090GRID',
                'SEA_ICE_VELOCITY_LLC0090GRID', 'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID',
                'SEA_ICE_SALT_PLUME_FLUX_LLC0090GRID', 'MIXED_LAYER_DEPTH_LLC0090GRID', 'OBP_LLC0090GRID',
                'OCEAN_VEL_LLC0090GRID', 'TEMP_SALINITY_LLC0090GRID', 'DENS_STRAT_PRESS_LLC0090GRID',
                'OCEAN_3D_VOLUME_FLUX_LLC0090GRID', 'OCEAN_3D_MOMENTUM_TEND_LLC0090GRID',
                'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID', 'OCEAN_3D_SALINITY_FLUX_LLC0090GRID', 'BOLUS_LLC0090GRID',
                'OCEAN_BOLUS_STREAMFUNCTION_LLC0090GRID']
datasets = pd.Series(
    ['GMSL_TIME_SERIES', 'GMSL_TIME_SERIES', 'GMSL_TIME_SERIES', 'GMAP_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES', 'SBO_CORE_TIME_SERIES',
     'SBO_CORE_TIME_SERIES', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID',
     'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID',
     'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID',
     'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID',
     'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID',
     'GEOMETRY_LLC0090GRID', 'GEOMETRY_LLC0090GRID', 'OCEAN_3D_MIX_COEFFS_LLC0090GRID',
     'OCEAN_3D_MIX_COEFFS_LLC0090GRID', 'OCEAN_3D_MIX_COEFFS_LLC0090GRID', 'SSH_LLC0090GRID', 'SSH_LLC0090GRID',
     'SSH_LLC0090GRID', 'SSH_LLC0090GRID', 'ATM_STATE_LLC0090GRID', 'ATM_STATE_LLC0090GRID', 'ATM_STATE_LLC0090GRID',
     'ATM_STATE_LLC0090GRID', 'ATM_STATE_LLC0090GRID', 'ATM_STATE_LLC0090GRID', 'STRESS_LLC0090GRID',
     'STRESS_LLC0090GRID', 'STRESS_LLC0090GRID', 'STRESS_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID',
     'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID',
     'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID',
     'HEAT_FLUX_LLC0090GRID', 'HEAT_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID',
     'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID',
     'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID', 'FRESH_FLUX_LLC0090GRID',
     'FRESH_FLUX_LLC0090GRID', 'SEA_ICE_CONC_THICKNESS_LLC0090GRID', 'SEA_ICE_CONC_THICKNESS_LLC0090GRID',
     'SEA_ICE_CONC_THICKNESS_LLC0090GRID', 'SEA_ICE_CONC_THICKNESS_LLC0090GRID', 'SEA_ICE_VELOCITY_LLC0090GRID',
     'SEA_ICE_VELOCITY_LLC0090GRID', 'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID', 'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID',
     'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID', 'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID',
     'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID', 'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID',
     'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID', 'SEA_ICE_HORIZ_VOLUME_FLUX_LLC0090GRID',
     'SEA_ICE_SALT_PLUME_FLUX_LLC0090GRID', 'SEA_ICE_SALT_PLUME_FLUX_LLC0090GRID', 'MIXED_LAYER_DEPTH_LLC0090GRID',
     'OBP_LLC0090GRID', 'OBP_LLC0090GRID', 'OBP_LLC0090GRID', 'OCEAN_VEL_LLC0090GRID', 'OCEAN_VEL_LLC0090GRID',
     'OCEAN_VEL_LLC0090GRID', 'TEMP_SALINITY_LLC0090GRID', 'TEMP_SALINITY_LLC0090GRID', 'DENS_STRAT_PRESS_LLC0090GRID',
     'DENS_STRAT_PRESS_LLC0090GRID', 'DENS_STRAT_PRESS_LLC0090GRID', 'DENS_STRAT_PRESS_LLC0090GRID',
     'OCEAN_3D_VOLUME_FLUX_LLC0090GRID', 'OCEAN_3D_VOLUME_FLUX_LLC0090GRID', 'OCEAN_3D_VOLUME_FLUX_LLC0090GRID',
     'OCEAN_3D_MOMENTUM_TEND_LLC0090GRID', 'OCEAN_3D_MOMENTUM_TEND_LLC0090GRID',
     'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID', 'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID',
     'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID', 'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID',
     'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID', 'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID',
     'OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID', 'OCEAN_3D_SALINITY_FLUX_LLC0090GRID',
     'OCEAN_3D_SALINITY_FLUX_LLC0090GRID', 'OCEAN_3D_SALINITY_FLUX_LLC0090GRID', 'OCEAN_3D_SALINITY_FLUX_LLC0090GRID',
     'OCEAN_3D_SALINITY_FLUX_LLC0090GRID', 'OCEAN_3D_SALINITY_FLUX_LLC0090GRID', 'OCEAN_3D_SALINITY_FLUX_LLC0090GRID',
     'OCEAN_3D_SALINITY_FLUX_LLC0090GRID', 'BOLUS_LLC0090GRID', 'BOLUS_LLC0090GRID', 'BOLUS_LLC0090GRID',
     'OCEAN_BOLUS_STREAMFUNCTION_LLC0090GRID', 'OCEAN_BOLUS_STREAMFUNCTION_LLC0090GRID'],
    index=all_variables)
timings = pd.Series(
    ['Daily', 'Snapshot', 'Snapshot', 'None', 'None', 'All', 'Daily', 'Daily', 'Daily', 'Daily', 'All', 'All', 'Daily',
     'Daily', 'Daily', 'All', 'Daily', 'All', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily'],
    index=all_datasets)
granule_prefixes = pd.Series(
    ['GLOBAL_MEAN_SEA_LEVEL', 'GLOBAL_MEAN_ATM_SURFACE_PRES', 'SBO_CORE_PRODUCTS', 'GRID_GEOMETRY',
     'OCEAN_3D_MIXING_COEFFS', 'SEA_SURFACE_HEIGHT', 'ATM_SURFACE_TEMP_HUM_WIND_PRES', 'OCEAN_AND_ICE_SURFACE_STRESS',
     'OCEAN_AND_ICE_SURFACE_HEAT_FLUX', 'OCEAN_AND_ICE_SURFACE_FW_FLUX', 'SEA_ICE_CONC_THICKNESS', 'SEA_ICE_VELOCITY',
     'SEA_ICE_HORIZ_VOLUME_FLUX', 'SEA_ICE_SALT_PLUME_FLUX', 'OCEAN_MIXED_LAYER_DEPTH', 'OCEAN_BOTTOM_PRESSURE',
     'OCEAN_VELOCITY', 'OCEAN_TEMPERATURE_SALINITY', 'OCEAN_DENS_STRAT_PRESS', 'OCEAN_3D_VOLUME_FLUX',
     'OCEAN_3D_MOMENTUM_TEND', 'OCEAN_3D_TEMPERATURE_FLUX', 'OCEAN_3D_SALINITY_FLUX', 'OCEAN_BOLUS_VELOCITY',
     'OCEAN_BOLUS_STREAMFUNCTION'],
    index=all_datasets)


def adjust_timing(variable: str, timing: str) -> str:
    dataset = datasets[variable]
    if timing not in {'None', 'Monthly', 'Daily', 'Monthly Snapshot', 'Daily Snapshot'}:
        raise ValueError(
            str(timing) + ' is not a valid timing (select either Monthly, Daily, Monthly Snapshot, or Daily Snapshot)')
    elif timing in {'Monthly Snapshot', 'Daily Snapshot'} and timings[dataset] == 'Daily':
        raise ValueError('No snapshots available for ' + str(variable))
    elif timing in {'Monthly', 'Daily'} and timings[dataset] == 'Snapshot':
        raise ValueError('No monthly or daily averages available for ' + str(variable))
    elif timings[dataset] == 'None':
        return 'None'
    elif timing == 'None' and timings[dataset] == 'Snapshot':
        return 'Monthly Snapshot'
    elif timing == 'None' and timings[dataset] in {'Daily', 'All'}:
        return 'Monthly'
    else:
        return timing


def get_granule(granule: str, directory: str) -> str:
    file = os.path.join(directory, os.path.basename(granule))
    if not os.path.isfile(file):
        with requests.get(url=granule) as r:
            if r.status_code == 401:
                raise IOError('Incorrect EarthData login; check netrc')
            elif r.status_code // 100 != 2:
                raise IOError(r.text)
            with open(file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
    return file


def ecco_dataset(dataset: str, start: dt.date = None, end: dt.date = None, timing: str = 'None'):
    short_timing_names = {'None': '', 'Monthly': '_MONTHLY', 'Daily': '_DAILY', 'Monthly Snapshot': '_SNAPSHOT',
                          'Daily Snapshot': '_SNAPSHOT'}
    long_timing_names = {'None': '', 'Monthly': '_mon_mean', 'Daily': '_day_mean', 'Monthly Snapshot': '_snap',
                         'Daily Snapshot': '_snap'}
    if timing not in short_timing_names:
        raise ValueError('Unrecognized timing: ' + str(timing))
    shortname = 'ECCO_L4_' + dataset + short_timing_names[timing] + '_V4R4'
    if 'LLC0090' in dataset:
        if timing == 'Monthly':
            start = dt.date(start.year, start.month, 1)
            dates = [date.strftime('_%Y-%m') for date in pd.date_range(start, end, freq='MS')]
        elif timing == 'Daily':
            dates = [date.strftime('_%Y-%m-%d') for date in pd.date_range(start, end)]
        elif timing in {'Monthly Snapshot', 'Daily Snapshot'}:
            dates = [date.strftime('_%Y-%m-%dT000000') for date in pd.date_range(start, end)]
        elif timing == 'None':
            dates = ['']
        longnames = [granule_prefixes[dataset] + long_timing_names[timing] + date + '_ECCO_V4r4_native_llc0090.nc'
                     for date in dates]
    else:
        longnames = [granule_prefixes[dataset] + long_timing_names[timing] + '_ECCO_V4r4_1D.nc']
    granules = ['https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/' + shortname + '/' + longname
                for longname in longnames]
    granule_dir = downloads + '/' + shortname
    try:
        os.mkdir(granule_dir)
    except FileExistsError:
        pass
    files = [get_granule(granule, granule_dir) for granule in granules]
    array = xr.open_mfdataset(files, data_vars='minimal', coords='minimal', compat='override')
    if timing == 'Monthly':
        times = pd.DatetimeIndex(array.time)
        array = array.assign_coords(time=[str(t)[:7] for t in times])
    elif timing in {'Daily', 'Daily Snapshot', 'Monthly Snapshot'}:
        times = pd.DatetimeIndex(array.time)
        array = array.assign_coords(time=[str(t)[:10] for t in times])
    if timing == 'Monthly Snapshot':
        array = array.sel(time=[t for t in array.time.values if t[8:10] == '01'])
        array = array.assign_coords(time=[t[:7] for t in array.time.values])
    return array


def ecco_variable(variable: str, start: dt.date | None = None,
                  end: dt.date | None = None, timing: str = 'None') -> xr.DataArray:
    if variable not in all_variables:
        raise ValueError(str(variable) + ' is not an ECCO variable')
    timing = adjust_timing(variable, timing)
    if timing != 'None' and start is None and 'LLC0090' in datasets[variable]:
        raise ValueError('Enter a date to retrieve \'' + str(variable) + '\'')
    if type(start) == str:
        if len(start) == 7:
            start += '-01'
        start = dt.datetime.strptime(start, '%Y-%m-%d')
    if type(end) == str:
        if len(end) == 7:
            end += '-01'
        end = dt.datetime.strptime(end, '%Y-%m-%d')
    if end is None:
        end = start
    return ecco_dataset(datasets[variable], start, end, timing)[variable]
