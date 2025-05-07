import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

from xspectra.simulation import *
from xspectra.utils import *
from xspectra.atom import *



def plot_spectra_with_rays(spectrum, longueurs_onde, rays, atoms= ["Ne I", "Ar I"], spectra_output=None):
    """
    Plots spectra with vertical lines representing atomic rays.
    Parameters:
    -----------
    spectrum : array-like
        The intensity values of the spectrum to be plotted.
    longueurs_onde : array-like
        The wavelength values corresponding to the spectrum.
    rays : list of list of float
        A list where each sublist contains the wavelengths of the rays for a specific atom.
    atoms : list of str, optional
        A list of atomic species corresponding to the rays. Default is ["Ne I", "Ar I"].
    spectra_output : str, optional
        File path to save the generated plot. If None, the plot is not saved.
    Returns:
    --------
    None
        Displays the plot and optionally saves it to a file.
    Notes:
    ------
    - Each subplot corresponds to a specific atom and its associated rays.
    - Vertical dashed red lines are drawn at the wavelengths of the rays.
    - The wavelength values of the rays are annotated on the plot.
    """
    
    fig, axs = plt.subplots(len(atoms), 1, figsize=(10, 2.5 * len(atoms)))

    for i, (atom, raies_atom) in enumerate(zip(atoms, rays)):
        axs[i].plot(longueurs_onde, spectrum)
        
        # Ajouter des lignes verticales pour les raies de chaque atome
        for ray in raies_atom:
            axs[i].axvline(x=ray, color='r', linestyle='--', alpha=0.7)
            axs[i].text(ray, 0.8, f"{ray:.1f}", rotation=90, verticalalignment='bottom')
        
        axs[i].set_xlabel("Longueur d'onde (nm)")
        axs[i].set_ylabel("Intensité")
        axs[i].set_title(f"Spectre de la lampe avec raies {atom}")

    plt.tight_layout()
    if spectra_output is not None:
        fig.savefig(spectra_output)
    plt.show()
    
    
def plot_scores_and_matches(atoms, scores, nb_matches, scores_output=None):
    """
    Plots a bar chart comparing scores and the number of matches for a list of atoms.
    The function sorts the atoms in descending order based on the number of matches, 
    then creates a dual-axis bar chart where one axis represents the scores and the 
    other represents the number of matches. The chart is displayed, and optionally 
    saved to a file.
    Args:
        atoms (list): A list of atom names or identifiers.
        scores (list): A list of scores corresponding to the atoms.
        nb_matches (list): A list of the number of matches corresponding to the atoms.
        scores_output (str, optional): File path to save the plot as an image. 
                                       If None, the plot is not saved. Defaults to None.
    Returns:
        None
    """
    
    # Trier les atomes par ordre décroissant des matchs
    sorted_indices = np.argsort(nb_matches)[::-1]
    sorted_atoms = [atoms[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_nb_matches = [nb_matches[i] for i in sorted_indices]

    # Configuration des barres
    x = np.arange(len(sorted_atoms))  # la position des atomes
    width = 0.35  # la largeur des barres

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Barres pour les scores
    bars1 = ax1.bar(x - width/2, sorted_scores, width, label='Score des raies', color='b')
    ax1.set_xlabel('Atomes')
    ax1.set_ylabel('Score des raies', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Créer un deuxième axe pour le nombre de matchs
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, sorted_nb_matches, width, label='Nombre de raies détectées', color='r')
    ax2.set_ylabel('Nombre de raies détectées', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Ajouter des labels, titre et légende
    ax1.set_title('Scores et nombre de raies détectées en fonction des atomes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_atoms)

    fig.tight_layout()
    if scores_output is not None:
        fig.savefig(scores_output)
        
    plt.show()

def show_result_calculation_Trot(wavelengths_target, spectrum_target, J_range=(8, 20), certainty=0.95, max_J=25, output_file=None, show=True):
    """
    Visualizes the results of rotational temperature (T_rot) calculation using a log method.
    This function plots the data points, the linear regression fit, and provides additional
    axes for quantum number (J) and wavelength (L). It also annotates the plot with the 
    calculated rotational temperature and its uncertainty.
    Parameters:
        spectrum_target (array-like): The target spectrum data.
        wavelengths_target (array-like): The corresponding wavelengths for the spectrum data.
        J_range (tuple, optional): The range of quantum numbers (J) to consider for the 
            linear regression. Defaults to (8, 20).
        certainty (float, optional): The confidence level for the uncertainty calculation. 
            Defaults to 0.95.
    Returns:
        None: The function generates a plot and does not return any value.
    Notes:
        - The function assumes that `get_points_log_method` and `get_reg_linear` are 
          defined elsewhere and provide the necessary data for plotting and regression.
        - The plot includes three synchronized x-axes: one for the calculated X values, 
          one for the quantum numbers (J), and one for the wavelengths (L).
        - The linear regression line is plotted along with its R² value and the calculated 
          rotational temperature (T_rot) with uncertainty.
    """
    J_deb, J_fin = J_range
    max_J = max(max_J, J_fin)

    J, L, X, Y = get_points_log_method(wavelengths_target, spectrum_target, max_J=max_J)
    
    T_rot, uT_rot, slope, uslope, intercept, r2, p_value, stderr = get_reg_linear(J, L, X, Y, J_range=J_range, certainty=certainty)
    
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Premier axe pour les données X et Y
    ax1.plot(X, Y, 'o', label="Data")
    ax1.set_xlabel(r"$B \cdot J \cdot (J + 1)$ en cm$^{-1}$")
    ax1.set_ylabel(r"$\log\left(I \cdot \lambda^4 / J\right)$")
    ax1.set_title("Data")
    ax1.grid()
    ax1.legend()

    # Deuxième axe pour les valeurs de J
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Synchroniser les limites des axes
    ax2.set_xticks(X)  # Positionner les ticks sur les valeurs de X
    ax2.set_xticklabels(J)  # Associer les ticks aux valeurs de J
    ax2.set_xlabel("Valeurs de J")

    # Troisième axe pour les longueurs d'onde
    ax3 = ax1.twiny()
    ax3.spines["top"].set_position(("axes", 1.2))  # Positionner au-dessus du deuxième axe
    ax3.set_xlim(ax1.get_xlim())  # Synchroniser les limites des axes
    ax3.set_xticks(X)  # Positionner les ticks sur les valeurs de X
    ax3.set_xticklabels(np.round(L, 2), rotation=45)  # Associer les ticks aux longueurs d'onde et incliner les étiquettes
    ax3.set_xlabel("Longueur d'onde (nm)")


    def get_index(array, value):
        """
        Returns the index of the element in the array that matches the given value.
        If the value is not found, raises a ValueError.
        """
        return np.where(array == value)[0][0]

    ax1.axvline(X[np.where(J == J_deb)[0][0]], color='purple', linestyle='--', alpha=0.7, label="limites régression linéaire")
    ax1.axvline(X[np.where(J == J_fin)[0][0]], color='purple', linestyle='--', alpha=0.7)

    # Trace la régression linéaire
    x_fit = np.linspace(X.min(), X.max(), 100)
    y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, 'r:', label="Régression linéaire", linewidth=2)
    ax1.legend()

    ax1.legend(title=f"R²: {r2:.4f}")
    ax1.annotate(f"T_rot = {T_rot:.0f} +/- {uT_rot:.0f} K", xy=(0.5, 0.9), xycoords="axes fraction", fontsize=12, ha="center", color='red')


    plt.tight_layout()
    if output_file is not None:
        fig.savefig(output_file)
        
    if show:
        plt.show()
    
    
    
    
def analyse_spectrum(filename, l_min=200, l_max=850, resolution=13.9072e-12, top=10, height=1e-2, cibles = ["Hg I", "H I", "He I", "Ne I", "Ar I", "Kr I", "Xe I", "Cd I", "Zn I", "Na I"], spectra_output=None, scores_output=None):
    """
    Analyse un spectre en détectant les raies et en les comparant aux raies cibles.

    Parameters:
    filename (str): Le nom du fichier contenant le spectre.
    l_min (float): La longueur d'onde minimale du spectre.
    l_max (float): La longueur d'onde maximale du spectre
    resolution (float): La résolution du spectre.
    top (int): Le nombre de raies à récupérer pour chaque atome.    
    height (float): La hauteur minimale des pics à détecter.
    """
    
    # Charger le spectre
    data = imageio.imread(filename)
    spectrum = np.sum(data, axis=0)
    spectrum /= max(spectrum) # Normalisation de l'intensité
    
    # Créer les longueurs d'onde correspondantes
    longueurs_onde = np.linspace(l_min, l_max, len(spectrum))
    
    # Extraire les raies du spectre
    peaks = extract_raies(spectrum, height=height)
    raies_detectees = longueurs_onde[peaks]
    
    
    print(f"Nombre de raies détectées: {len(raies_detectees)}")
    
    # Récupérer les raies cibles
    rays = [get_best_rays(atom, l_min=l_min, l_max=l_max, top=top)[0] for atom in cibles]
    
    # Calculer les scores et le nombre de matchs
    scores = [score_rays(raies_detectees, raies_atom) for raies_atom in rays]
    nb_matches = [get_nb_matches(raies_detectees, raies_atom) for raies_atom in rays]
    
    print(f"Scor Hg " + str(scores[0]))
    print(f"Nb match Hg " + str(nb_matches[0]))
    
    # Affichage des résultats
    plot_spectra_with_rays(spectrum, longueurs_onde, rays, cibles, spectra_output)
    plot_scores_and_matches(cibles, scores, nb_matches, scores_output)
    
    return scores, nb_matches


