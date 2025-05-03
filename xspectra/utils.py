import numpy as np
import imageio.v2 as imageio
from xspectra.simulation import *
from scipy import constants as const
import os


# Traitons les valeurs pour ne garder que les valeurs 5%-95%
def filter_data(data, v=5):
    q05, q95 = np.percentile(data, [v, 100-v])
    return np.clip(data, q05, q95)

def load_data(file):
    data = imageio.imread(file)
    return data

def compute_spectra(data, normalize=True):
    spectrum = np.sum(data, axis=0)
        
    if normalize:
        spectrum = spectrum / np.max(spectrum)
        
    return spectrum

def get_spectra(file, cut_proportion=0.75, filter_data_value=5, normalize=True):
    data = imageio.imread(file)
    
    if filter_data_value is not None:
        data = filter_data(data, filter_data_value)
        
        
    if cut_proportion is not None: 
        data = cut_data(data, cut_proportion)
    
    return compute_spectra(data, normalize)

def borne_spectra(spectra, borne=5):
    m, M = np.min(spectra), np.percentile(spectra, [100-borne])
    return np.clip(spectra, m, M)

def get_maxima_image(filename):
    data = load_data(filename)
    return np.max(data)

def get_quantile_image(filename, q=0.95):
    """
    Get the q-quantile
    """
    data = load_data(filename)
    return np.quantile(data, q)

def cut_data(*args, **kwargs):
    """
    Cut the data on a middle bandwidth
    Define :
    
    propotion: proportion of the data to keep
    
    or a tuple of two values to define the cuts
    
    custs : tuple of two values to define the cuts
    
    """
    data = args[0]
    if "cuts" in kwargs:
        return data[kwargs["cuts"][0]:kwargs["cuts"][1]]
    else:
        proportion = kwargs.get('proportion', 0.75)
        cuts = int(data.shape[0] * (1-proportion)/2), int(data.shape[0] * (1+proportion)/2)
        return data[cuts[0]:cuts[1],:]
    
    
def get_winspec_spectra(filename="../data/2025-03-28-capillary-discharge/spectra-337.txt"):
    import pandas as pd
    data = pd.DataFrame()
    with open(filename, "r") as file:
        for _ in range(3):
            line = file.readline().strip().split("\t")
            data[line[0]] = line[1:]
        
    return data


def get_points_folder(folder, method="maxima", kyrill=False):
    files = os.listdir(folder)
    delays = []
    values = []
    for file in sorted(files):
        if file.endswith(".SPE"):
            if kyrill :
                delay = file[-9:-6]
            else:
                delay = file.split("-")[-1].split(".")[0][1:]
            # print(delay)
            delays.append(int(delay))
            values.append(load_data(os.path.join(folder, file)))
            
    return np.array(delays), np.array(values)


def get_temp_vib(r):
    return 3.954e-20 / (const.k * np.log(r * (337.09546/333.81134)))

def transform(X, x1=337.459, y1=337.095, x2=334.44, y2=333.81):
    """
    Transforme les coordonnées (x1, y1) en (x2, y2)
    """
    return (y2-y1) / (x2-x1) * (X - x1) + y1

#### Calcul de T_vib ######

def calculate_temperature_single(spectrum, i1=529, i2=169):
    spectrum = spectrum.copy()
    # spectrum -= min(spectrum)
    r = spectrum[i1] / spectrum[i2]
    return get_temp_vib(r)


#### METHODE FIT SPECTRE THEORIQUE #######

def compute_score_fit(spectrum, fit):
    """
    Compute the score of the fit
    """
    return np.sum((spectrum - fit)**2) 

def trichotomy(f, a, b, tol=1e-5):
    """
    Trichotomy method to find the minimum of a function
    """
    while (b - a) > tol:
        c = a + (b - a) / 3
        d = a + 2 * (b - a) / 3
        if f(c) < f(d):
            b = d
        else:
            a = c
    return (a + b) / 2 


def get_best_fit(wavelengths, spectrum, T_el=1_000, T_vib=907, T_rot=1_000, elargissement=0.1, w_decalage=0, T_range=(100, 10_000), elargissement_range=(0.01, 5), w_decalage_range=(-2,2), verbose=False, nb_steps=10):
    """
    Get the best fit for the spectrum
    On suppose que les fonctions sont convexes... pour la trichotomie
    """
    T_min, T_max = T_range
    min_elargissement, max_elargissement = elargissement_range
    min_w_decalage, max_w_decalage = w_decalage_range
    
    # On cherche la température électronique et la taille de la fente
    for i in range(nb_steps):
        T_rot = trichotomy(lambda x: compute_score_fit(spectrum, get_spectrum(wavelengths+w_decalage, T_el=T_el, T_vib=T_vib, T_rot=x)), T_min, T_max)
        elargissement = trichotomy(lambda x: compute_score_fit(spectrum, get_spectrum(wavelengths+w_decalage, T_el=T_el, T_vib=T_vib, T_rot=T_rot, sigma_exp=x)), min_elargissement, max_elargissement)
        w_decalage = trichotomy(lambda x: compute_score_fit(spectrum, get_spectrum(wavelengths+x, T_el=T_el, T_vib=T_vib, T_rot=T_rot, sigma_exp=elargissement)), min_w_decalage, max_w_decalage)
        
        score = compute_score_fit(spectrum, get_spectrum(wavelengths, T_el=T_el, T_rot=T_rot))
        if verbose:
            print(f"Iteration {i+1:3d} | Score: {score:8.3f} | Elargissement: {elargissement:6.2f} nm | T_rot: {T_rot:6.0f} K | Décalage: {w_decalage:6.2f} nm")
    
    return score, T_rot, elargissement, w_decalage

def get_best_fit_random(wavelengths, spectrum, T_el=1_000, T_vib=1_000, T_rot=1_000, n=100, T_range=(100, 10_000), verbose=False):
    """
    Get the best fit for the spectrum
    On suppose que les fonctions sont convexes... pour la trichotomie
    """
    T_min, T_max = T_range
    # On cherche la température rotationnelle
    best_score = np.inf
    best_T_el = T_el
    best_T_rot = T_rot
    
    for i in range(n):
        T_rot = np.random.uniform(T_min, T_max)
        score = compute_score_fit(spectrum, get_spectrum(wavelengths, T_el=T_el, T_vib=T_vib, T_rot=T_rot))
        if score < best_score:
            best_score = score
            best_T_el = T_el
            best_T_rot = T_rot
            if verbose:
                print(f"Iteration {i+1} : Score {score} \t T_el = {T_el:.0f} K \t T_rot = {T_rot:.0f} K")
    
    return best_score, best_T_rot

# get_best_fit_random(filtered_spectrum_target, filtered_wavelengths_target, T_el=1_000, T_rot=1_000, n=100)


#### METHODE ANALYTIQUE BRANCHE R ######## 

def get_value(x, X, Y, method="nearest", nb_points=5):
    """
    Get the value of the function at x
    """
    
    # ou moyenne autour  ???????
    # On doit prendre la valeur la plus proche de x dans X
    if method == "nearest":
        # On prend la valeur la plus proche
        index = np.abs(X - x).argmin()
        return Y[index]
    elif method == "linear":
        # On interpole linéairement
        index = np.abs(X - x).argmin()
        if index == 0:
            return Y[0]
        elif index == len(X) - 1:
            return Y[-1]
        else:
            return np.interp(x, X[index-1:index+1], Y[index-1:index+1])
    elif method == "mean":
        # on fait la moyenne des n points autour de x
        index = np.abs(X - x).argmin()
        return np.mean(Y[max(index-nb_points//2,0):min(index+nb_points//2,len(Y))])
    else:
        raise ValueError("Method must be 'nearest' or 'linear'")
    
def get_points_log_method(wavelengths_target, spectrum_target, max_J=25):
    """
    Computes the logarithmic intensity values and associated parameters for a given spectrum.
    This function calculates the energy difference between two states, converts it to a wavelength,
    and computes the logarithmic intensity values for a range of rotational quantum numbers (J).
    The results are used to analyze spectral data.
    Parameters:
    -----------
    wavelengths_target : numpy.ndarray
        Array of target wavelengths (in nm) for the spectrum.
    spectrum_target : numpy.ndarray
        Array of intensity values corresponding to the target wavelengths.
    Returns:
    --------
    J : numpy.ndarray
        Array of rotational quantum numbers.
    L : numpy.ndarray
        Array of calculated wavelengths (in nm) corresponding to the energy differences.
    X : numpy.ndarray
        Array of X values, calculated as B_C3P * J * (J + 1), where B_C3P is a constant.
    Y : numpy.ndarray
        Array of logarithmic intensity values, calculated as log(intensity * (wavelength^4) / J).
    Notes:
    ------
    - The function assumes the existence of `energy_function_C3P`, `energy_function_B3P`, and `get_value` 
      functions, which are used to compute energy levels and retrieve intensity values.
    - The constant `B_C3P` must be defined in the global scope for the function to work correctly.
    - The method parameter in `get_value` is set to "mean" for intensity calculation.
    """
    # On récupère les points pour les différentes valeurs de J pour la bande R : J2 = j-1

    J = np.arange(1, max_J+1)
    L = np.zeros(len(J))
    X = np.zeros(len(J))
    Y = np.zeros(len(J))

    v1, v2 = 0,0

    for i, j in enumerate(J):
        j1, j2 = j, j-1
        
        _,_,_,E_1 = energy_function_C3P(v=v1, j=j1)
        _,_,_,E_2 = energy_function_B3P(v=v2, j=j2)
        Delta_E = E_1 - E_2 # cm-1
        l_true = 1/Delta_E * (1e7) # (nm)  longueur d'onde
        
        intensity = get_value(l_true, wavelengths_target, spectrum_target, method="mean") # get the intensity
        # print(intensity)
        
        L[i] = l_true
        Y[i] = np.log(intensity * (l_true**4) / j)
        X[i] = B_C3P * j * (j+1) # cm-1
        
    return J, L, X, Y
        
        
def get_reg_linear(J, L, X, Y, J_range=(8, 20), certainty=0.95):
    """
    Perform a linear regression on a subset of data points and calculate related statistics.
    This function filters the input data based on the specified range of `J` values,
    performs a linear regression on the filtered data, and computes the rotational
    temperature and its uncertainty.
    Parameters:
    -----------
    J : array-like
        Array of J values (e.g., rotational quantum numbers).
    L : array-like
        Array of L values (not used in the current implementation).
    X : array-like
        Array of x-coordinates for the data points.
    Y : array-like
        Array of y-coordinates for the data points.
    J_range : tuple, optional
        A tuple specifying the range of `J` values to include in the regression
        (default is (8, 20)).
    certainty : float, optional
        The confidence level for calculating the uncertainty on the slope
        (default is 0.95).
    Returns:
    --------
    T_rot : float
        The rotational temperature (in Kelvin) calculated from the slope of the regression.
    uT_rot : float
        The uncertainty in the rotational temperature (in Kelvin).
    slope : float
        The slope of the linear regression.
    uslope : float
        The uncertainty in the slope of the linear regression.
    intercept : float
        The y-intercept of the linear regression.
    r2 : float
        The coefficient of determination (R-squared) of the regression.
    p_value : float
        The p-value for the hypothesis test on the slope.
    stderr : float
        The standard error of the regression.
    Notes:
    ------
    - The function uses the `scipy.stats.linregress` method for linear regression.
    - The uncertainty on the slope is calculated using the Student's t-distribution.
    - The rotational temperature is derived from the slope using a custom conversion
      function `cm2J` (not provided in this code snippet).
    """
    # Tracer des lignes verticales pour j_in = 8 et J_max = 20
    # Tracer des lignes verticales pour j_in = 8 et J_max = 20
    J_deb, J_fin = J_range

    mask_J = (J >= J_deb) & (J <= J_fin)

    X_reg, Y_reg = X[mask_J], Y[mask_J]

    # Régression linéaire
    from scipy.stats import linregress # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

    linreg_result = linregress(X_reg, Y_reg)

    slope = linreg_result.slope
    intercept = linreg_result.intercept
    r_value = linreg_result.rvalue
    r2 = linreg_result.rvalue**2
    p_value = linreg_result.pvalue
    stderr = linreg_result.stderr


    # Calcul de l'incertitude sur la pente
    from scipy.stats import t
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(1-certainty, len(X_reg)-2)
    uslope = ts*linreg_result.stderr

    T_rot = cm2J(- 1 / slope) / const.k # K
    uT_rot = - T_rot * uslope / slope # K
    
    return T_rot, uT_rot, slope, uslope, intercept, r2, p_value, stderr

def compute_Trot_with_branch(wavelengths_target, spectrum_target, J_range=(8, 20), certainty=0.95, max_J=25):
    """
    Compute the rotational temperature using the R branch of the spectrum.
    Parameters:
    -----------
    spectrum_target : array-like
        The target spectrum data.
    wavelengths_target : array-like
        The target wavelengths corresponding to the spectrum.
    J_range : tuple, optional
        A tuple specifying the range of `J` values to include in the regression
        (default is (8, 20)).
    certainty : float, optional
        The confidence level for calculating the uncertainty on the slope
        (default is 0.95).
    Returns:
    --------
    T_rot : float
        The rotational temperature (in Kelvin) calculated from the slope of the regression.
    uT_rot : float
        The uncertainty in the rotational temperature (in Kelvin).
    slope : float
        The slope of the linear regression.
    uslope : float
        The uncertainty in the slope of the linear regression.
    intercept : float
        The y-intercept of the linear regression.
    r2 : float
        The coefficient of determination (R-squared) of the regression.
    p_value : float
        The p-value for the hypothesis test on the slope.
    stderr : float
        The standard error of the regression.
    """
    J_deb, J_fin = J_range
    max_J = max(max_J, J_fin)
    
    J, L, X, Y = get_points_log_method(wavelengths_target, spectrum_target, max_J=max_J)
    
    T_rot, uTrot, _, _, _, _, _, _=  get_reg_linear(J, L, X, Y, J_range=J_range, certainty=certainty)
    
    return T_rot, uTrot


### Extraction des pics

def extract_raies(spectrum, height=1e-2):
    """
    Extrait les raies d'un spectre donné.

    Parameters:
    spectrum (numpy.ndarray): Le spectre à analyser.
    height (float): La hauteur minimale des pics à détecter.

    Returns:
    numpy.ndarray: Les indices des raies détectées dans le spectre.
    """
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(spectrum, height=height)
    return peaks


def closest_peak_distance(raie, peaks):
    """
    Calcule la distance entre une raie donnée et le pic le plus proche.

    Parameters:
    raie (float): La longueur d'onde de la raie.
    peaks (numpy.ndarray): Les longueurs d'onde des pics détectés.

    Returns:
    float: La distance à la raie la plus proche.
    """
    return np.min(np.abs(peaks - raie))


def delete_background(data, zone_background=(338, 341)):
    """
    Removes the background signal from the provided data and normalizes the result.
    This function calculates the mean value of the signal within a specified background zone
    and subtracts it from the signal. The resulting data is then normalized by dividing 
    all values by the maximum value in the signal.
    Parameters:
    -----------
    data : numpy.ndarray
        A 2D array where the first column represents the x-axis (e.g., energy or wavelength)
        and the second column represents the y-axis (e.g., intensity or signal).
    zone_background : tuple of two floats, optional
        A tuple specifying the range (min, max) of the background zone on the x-axis.
        The default is (338, 341).
    Returns:
    --------
    numpy.ndarray
        A 2D array with the same shape as the input `data`, where the background signal
        has been removed and the data has been normalized.
    """
    
    data = data.copy()
    filtre_zone_background = (zone_background[0] < data[:,0]) & (data[:,0] < zone_background[1])
    background = data[filtre_zone_background, 1].mean()
    data[:, 1] -= background
    
    # Normalisation
    data[:, 1] /= data[:, 1].max()
    return data


def find_index_primary_peaks(wavelengths, spectrum, height=0.01, distance=100, w_sec_peak=333.82):
    """Find the indices of the primary and secondary peaks in a spectrum.

    Args:
        wavelengths (np.ndarray): Array of wavelengths corresponding to the spectrum.
        spectrum (np.ndarray): Array of intensity values representing the spectrum.
        height (float, optional): Minimum height of the peaks. Defaults to 0.01.
        distance (int, optional): Minimum distance between peaks. Defaults to 100.
        w_sec_peak (float, optional): Wavelength of the secondary peak. Defaults to 333.82.

    Returns:
        tuple: Indices of the primary and secondary peaks in the spectrum.
    """
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(spectrum, height=height, distance=distance)
    primary_peak = np.argmax(spectrum)
    secondary_peak = peaks[np.abs(wavelengths[peaks]-w_sec_peak).argmin()] # La longueur d'onde de la transition v'=1->v''=1 est de 333.82 nm
    return primary_peak, secondary_peak



