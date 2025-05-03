from astroquery.nist import Nist
import astropy.units as u
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import os

from xspectra.utils import *
from xspectra.atom import *


def get_best_rays(atom="Hg I", l_min=200, l_max=850, top=10):
    """
    Retrieve the best spectral lines for a given atom within a specified wavelength range.
    This function queries the NIST Atomic Spectra Database for spectral lines of a specified atom
    within a given wavelength range. It filters and sorts the results based on relative intensity
    and returns the top transitions.
    Parameters:
        atom (str): The atomic species to query (e.g., "Hg I" for neutral mercury). Default is "Hg I".
        l_min (float): The minimum wavelength in nanometers for the query range. Default is 200 nm.
        l_max (float): The maximum wavelength in nanometers for the query range. Default is 850 nm.
        top (int): The number of top transitions to return based on relative intensity. Default is 10.
    Returns:
        tuple: A tuple containing two numpy arrays:
            - The first array contains the observed wavelengths of the top transitions.
            - The second array contains the corresponding relative intensities of the transitions.
    Notes:
        - The function removes rows where the "Rel." column cannot be converted to a float.
        - The "Rel." column is used to sort the transitions by relative intensity in descending order.
    """
    
    # On récupère la table
    data_table = Nist.query(l_min*u.nm, l_max*u.nm, linename=atom, wavelength_type='vac+air').to_pandas()
    # Remove rows where "Rel." cannot be converted to float
    data_table = data_table[pd.to_numeric(data_table['Rel.'], errors='coerce').notnull()]
    data_table["Rel."] = pd.to_numeric(data_table["Rel."])
    # On trie par ordre décroissant
    data_table = data_table.sort_values(by='Rel.', ascending=False)
    # On prend les top transitions
    data_table = data_table.head(top)
    return (np.array(data_table["Observed"]), np.array(data_table["Rel."]))




    


def score_rays(observed, target):
    """
    Calcule le score des raies observées par rapport aux raies cibles.

    Parameters:
    observed (numpy.ndarray): Les longueurs d'onde des raies observées.
    target (numpy.ndarray): Les longueurs d'onde des raies cibles.

    Returns:
    float: Le score des raies observées.
    """
    return np.mean([closest_peak_distance(raie, observed)**2 for raie in target]) # distance moyenne au plus proche

def get_nb_matches(observed, target, threshold=2):
    """
    Calcule le nombre de raies observées qui correspondent à des raies cibles.

    Parameters:
    observed (numpy.ndarray): Les longueurs d'onde des raies observées.
    target (numpy.ndarray): Les longueurs d'onde des raies cibles.
    threshold (float): La distance maximale pour considérer une raie comme détectée.

    Returns:
    int: Le nombre de raies détectées.
    """
    return np.sum([closest_peak_distance(raie, observed) < threshold for raie in target])



