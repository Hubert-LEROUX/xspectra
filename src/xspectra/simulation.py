import numpy as np
import scipy as sp
from scipy import constants as const


# Load some data

# Constantes cf Ochkin
T_C3P = 89136.9 # cm-1 
T_B3P = 59619.3 # cm-1

omega_e_C3P = 2047.17 # cm^-1
omega_e_B3P = 1733.39 # cm^-1

omega_chi_C3P = 28.445 # cm^-1
omega_chi_B3P = 14.122 # cm^-1

# Influence du rotationnel
B_C3P = 1.825
B_B3P = 1.637


franck_condon_factors = [

    [4.55e-1, 3.31e-1, 1.45e-1, 4.94e-2, 1.45e-2, 3.87e-3, 9.68e-4, 2.31e-4, 5.36e-5, 1.21e-5, 2.61e-6, 5.23e-7, 9.1e-8],

    [3.88e-1, 2.29e-2, 2.12e-1, 2.02e-1, 1.09e-1, 4.43e-2, 1.52e-2, 4.68e-3, 1.33e-3, 3.57e-4, 9.15e-5, 2.25e-5, 5.22e-6],

    [1.34e-1, 3.35e-1, 2.3e-2, 6.91e-2, 1.69e-1, 1.41e-1, 7.72e-2, 3.32e-2, 1.23e-2, 4.12e-3, 1.27e-3, 3.69e-4, 1.03e-4],

    [2.16e-2, 2.52e-1, 2.04e-1, 8.81e-2, 6.56e-3, 1.02e-4, 1.37e-1, 9.93e-2, 5.26e-2, 2.31e-2, 8.95e-3, 3.16e-3, 1.03e-3],

    [1.15e-3, 5.66e-2, 3.26e-1, 1.13e-1, 1.16e-1, 2.45e-3, 4.7e-2, 1.09e-1, 1.04e-1, 6.67e-2, 3.4e-2, 1.5e-2, 5.97e-3],

]


franck_condon_factors = np.array(franck_condon_factors)



def cm2J(x):
    return x*sp.constants.h*sp.constants.c*10**2

def get_energy_function_given_state(omega_e=omega_e_B3P, omega_chi=omega_chi_B3P, T=T_B3P, B=B_B3P):
    def energy_function(v=0, j=0, in_cm=True):
        # TODO ajouter la correction au deuxième ordre pour le rotationnel
        T_vib = omega_e*(v+0.5) - omega_chi*(v+0.5)**2
        T_rot = B*(j*(j+1))
        if in_cm:
            return (T, T_vib, T_rot, T + T_vib + T_rot)
        return (cm2J(T), cm2J(T_vib), cm2J(T_rot), cm2J(T + T_vib + T_rot))
    return energy_function

energy_function_B3P = get_energy_function_given_state(omega_e=omega_e_B3P, omega_chi=omega_chi_B3P, T=T_B3P, B=B_B3P)
energy_function_C3P = get_energy_function_given_state(omega_e=omega_e_C3P, omega_chi=omega_chi_C3P, T=T_C3P, B=B_C3P)

def get_botzmann_population_distribution(T_e=T_B3P, omega_e=omega_e_B3P, omega_chi=omega_chi_B3P, B=B_B3P):
    """
    Return a function to compute the population distribution of the vibrational levels of the specified electronic state
    Parameters
    ----------
    T_e : float
        Energy of the electronic state in cm^-1
    omega_e : float
        Vibrational constant in cm^-1
    omega_chi : float
        Anharmonicity constant in cm^-1
    B : float
        Rotational constant in cm^-1

    Returns
    -------
    function
        A function that computes the Boltzmann population distribution for a given vibrational (v) and rotational (J) level
    ```
    """
    def boltzmann_population_distribution_given_state(sigma=2, S=1, L=1, v=0, J=0, T_el=1_000, T_rot=1_000, T_vib=1_000):
        """
        Computes the probabilty of population of an energy level given the temperature of the electronic, rotational and vibrational states
        T_e : float
            temperature of the electronic state in K
        v : int
            vibrational level
        sigma : int
            2 for homonuclear molecules, 1 for heteronuclear molecules
        S : float
            Spin multiplicity
        L : int
            Lambda quantum number
        J : int
            rotational level    
        T_el : float        
            temperature of the electronic state in K
        T_rot : float
            temperature of the rotational state in K
        T_vib : float
            temperature of the vibrational state
            
        This function is for B3P state only
        """
        #! TODO Ajouter L(J, P, P') dans la formule

        E_e, E_vib, E_rot, _ = energy_f(v=v, j=J, in_cm=False) # on veut en joules
        # g_e = (2-delta(0,Lambda))(2S+1)-> dégénérescence du niveau électronique
        g_e = 2 * (2*S+1) if L !=0 else 2*S+1
        
        return g_e*(2*J+1)*np.exp(-E_e/(const.k*T_el)) * np.exp(-E_vib/(const.k*T_vib)) * np.exp(-E_rot/(const.k*T_rot)) / sigma
    
    energy_f = get_energy_function_given_state(omega_e=omega_e, omega_chi=omega_chi, T=T_e, B=B)
    return boltzmann_population_distribution_given_state

population_distribution_B3P = get_botzmann_population_distribution(T_e=T_B3P, omega_e=omega_e_B3P, omega_chi=omega_chi_B3P, B=B_B3P)
population_distribution_C3P = get_botzmann_population_distribution(T_e=T_C3P, omega_e=omega_e_C3P, omega_chi=omega_chi_C3P, B=B_C3P)

def get_hl_factor(L1, L2, J1, J2):
    """
    Get the Hönl-London factor for the transition between two states

    Parameters
    ----------
    L1 : int
        Lambda quantum number of the upper state
    L2 : int
        Lambda quantum number of the lower state
    J1 : int        
        Rotational level of the upper state
    J2 : int    
        Rotational level of the lower state

    Returns
    -------
    float
        Hönl-London factor

    Raises
    ------
    ValueError
        If the input quantum numbers are invalid.
    """
    # Input validation:
    if not all(isinstance(x, int) for x in [L1, L2, J1, J2]):
        raise ValueError("All quantum numbers must be integers.")

    if L2 == L1:
        if J2 == J1:  # Q branch
            if J1 == 0:
                raise ValueError("J1 cannot be 0 for this transition.")  # Handle division by zero explicitly
            return (L1**2 * (2*J1+1)) / (J1 * (J1+1))
        elif J2 == J1 - 1:  # R branch
            return (J1+L1)*(J1-L1)/J1
        elif J2 == J1 + 1:  # P branch
            return (J1+L1+1)*(J1-L1+1)/(J1+1)
        raise ValueError("Invalid quantum numbers for transition with L2 == L1.")

    elif L2 == L1 + 1:
        if J2 == J1:  # Q branch
            return ((J1-L1)*(J1+L1+1)*(2*J1+1)) / (4*J1*(J1+1))
        elif J2 == J1 - 1:  # R branch
            return (J1+L1)*(J1+L1+1)/(4*(J1+1))
        elif J2 == J1 + 1:  # P branch
            return (J1+L1+1)*(J1+L1+2)/(4*(J1+1))
        raise ValueError("Invalid quantum numbers for transition with L2 == L1 + 1.")

    elif L2 == L1 - 1:
        if J2 == J1 - 1:  # R branch
            return (J1-L1)*(J1-L1+1)/(4*(J1+1))
        elif J2 == J1 + 1:  # P branch
            return (J1-L1+1)*(J1-L1+2)/(4*(J1+1))
        raise ValueError("Invalid quantum numbers for transition with L2 == L1 - 1.")

    print(L1, L2, J1, J2)
    raise ValueError("Invalid quantum numbers for transition.")

def get_A(l, L1, L2, v1, v2, J1, J2):
    """
    
    Get the Einstein A coefficient for the transition between two states
    Parameters
    ----------
    l : float
        Wavelength of the transition in nm
    L1 : int
        Lambda quantum number of the upper state
    L2 : int
        Lambda quantum number of the lower state
    v1 : int
        Vibrational level of the upper state
    v2 : int
        Vibrational level of the lower state
    J1 : int        
        Rotational level of the upper state
    J2 : int    
        Rotational level of the lower state
    """
    # get frenquency of the transition
    f = const.c / l # in Hz
    # TODO ajouter le facteur 2S+1 
    #  le facteur 2-delta(0,Lambda)
    delta_factor = 2 if L1 == 0 else 1
    g_el = delta_factor * (2*1+1) # poids statistique du niveau A_2 (Z, m) #* je sais pas du tout ce que c'est
    R = 1 # moment de transition electronique #* il faudrait trouver une formule pour ça
    fc_factor = franck_condon_factors[v1,v2] #* facteur de Franck Condon

    hl_factor = get_hl_factor(L1, L2, J1, J2) #* facteur de hönl-london

    return (64 * (np.pi ** 4 ) * (f ** 3) ) / (3*const.h*const.c**3*g_el*(2*J1+1)) * R**2 * fc_factor * hl_factor 
    # return R**2 * fc_factor / l**3 




def get_transition_probability(l, n_1=1, L1=1, L2=1, v1=0, v2=0, J1=0, J2=0):
    """
    Get's the probability of transition between two levels for a given wavelength l
    Parameters
    ----------
    l : float
        Wavelength of the transition in nm
    n_1 : int in mol
        Quantity of matter in the upper level
    A : float
        Transition constant (Einstein coefficient) in s^-1
    """
    A = get_A(l, L1, L2, v1, v2, J1, J2)
    return n_1 * (A/(4*np.pi)) * const.h * const.c / l

def gaussian(X, mu, sigma=0.1):
    return np.exp(-((X - mu) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def trapzezoidal(X, mu, width_top=0.22, width_bottom=0.39, intensity=2):
    """
    """
    assert width_top < width_bottom, "width_top must be smaller than width_bottom"
    # On fait un trapèze centré sur mu
    pente = 2*(intensity) / (width_bottom - width_top)
    return np.clip(pente*width_bottom/2- pente * np.abs(X - mu), 0, intensity)

def creneau(X, mu, width=0.22, intensity=2):
    """
    """
    return intensity * (np.abs(X - mu) < width/2)

def lorentzian(X, mu, gamma=0.1):
    return (gamma / np.pi) / ((X - mu) ** 2 + gamma ** 2)

def voigt(X, mu, sigma=0.1, gamma=0.1):
    """
    Voigt profile
    """
    # TODO : implementer la fonction de voigt
    return gaussian(X, mu, sigma) * lorentzian(X, mu, gamma)


    
def get_spectrum(wavelengths, v1=0, v2=0, nb_rot_levels=60,  T_el=300, T_rot=300, T_vib=300, sigma_exp=0.1, shape=gaussian):
    # wavelengths = np.linspace(llims[0], llims[1], nb_points)
    spectrum = np.zeros(len(wavelengths), dtype=float)

    def get_l_prob(v1, j1, v2, j2):
        _,_,_,E_1 = energy_function_C3P(v=v1, j=j1)
        _,_,_,E_2 = energy_function_B3P(v=v2, j=j2)
        Delta_E = E_1 - E_2 # cm-1
        l = 1/Delta_E * (1e7) # (nm)  longueur d'onde
        n_1 = population_distribution_B3P(v=v1, J=j1, T_el=T_el, T_rot=T_rot, T_vib=T_vib)/ (l*10**9)

        return l, get_transition_probability(l, n_1=n_1, L1=1, L2=1, v1=v1, v2=v2, J1=j1, J2=j2)


    for i in range(0, nb_rot_levels):
        if i-1 >= 0:
            j1, j2 = i, i-1 #! Raie P
            l, p = get_l_prob(v1, j1, v2, j2)
            spectrum += shape(wavelengths, l, sigma_exp) * p * wavelengths

            
        if i+1 < nb_rot_levels:
            j1, j2 = i, i+1 #! Raie R   
            l, p = get_l_prob(v1, j1, v2, j2)
            spectrum += shape(wavelengths, l, sigma_exp) * p * wavelengths

        if i > 0:   
            j1, j2 = i, i
            l, p = get_l_prob(v1, j1, v2, j2)#! Raie Q
            spectrum += shape(wavelengths, l, sigma_exp) * p * wavelengths
    return spectrum/np.max(spectrum)


def dichotomy_search(func, x0, x1, tol=1e-5):
    """
    Perform a dichotomy search to find the root of a function within a specified interval.

    Parameters
    ----------
    func : callable
        The function for which the root is to be found.
    x0 : float
        The lower bound of the interval.        
    x1 : float
        The upper bound of the interval.
        
    tol : float, optional
        The tolerance for convergence. Default is 1e-5.
    Returns
    -------
    float
        The root of the function within the specified interval.
    Raises
    ------
    ValueError
        If the function does not change sign over the interval.
    """
    # Check if the function changes sign over the interval
    if func(x0) * func(x1) > 0:
        raise ValueError("Function does not change sign over the interval.")

    while abs(x1 - x0) > tol:
        mid = (x0 + x1) / 2
        if func(mid) == 0:
            return mid
        elif func(mid) * func(x0) < 0:
            x1 = mid
        else:
            x0 = mid

    return (x0 + x1) / 2

def get_whole_spectrum(wavelengths, nb_vib_levels=5, nb_rot_levels=60,  T_el=300, T_rot=300, T_vib=300, sigma_exp=0.1, shape=gaussian, relative_addition=None):
    """
    Compute the whole spectrum for a given range of wavelengths based on vibrational and rotational transitions.
    Parameters:
        wavelengths (numpy.ndarray): Array of wavelengths (in nm) over which the spectrum is computed.
        nb_vib_levels (int, optional): Number of vibrational levels to consider. Default is 5.
        nb_rot_levels (int, optional): Number of rotational levels to consider. Default is 60.
        T_el (float, optional): Electronic temperature in Kelvin. Default is 300.
        T_rot (float, optional): Rotational temperature in Kelvin. Default is 300.
        T_vib (float, optional): Vibrational temperature in Kelvin. Default is 300.
        sigma_exp (float, optional): Standard deviation for the line shape function. Default is 0.1.
        shape (callable, optional): Line shape function (e.g., Gaussian). Must accept arguments (wavelengths, center, sigma). Default is `gaussian`.
    Returns:
        numpy.ndarray: Normalized spectrum array corresponding to the input wavelengths.
    Notes:
        - The function calculates the spectrum by iterating over vibrational and rotational levels.
        - It considers P, Q, and R branches of transitions and computes the corresponding probabilities and wavelengths.
        - The spectrum is normalized by dividing by its maximum value.
        - The `energy_function_C3P`, `energy_function_B3P`, `population_distribution_B3P`, and `get_transition_probability` functions are assumed to be defined elsewhere in the codebase.
    """
    wmin, wmax = wavelengths[0], wavelengths[-1]    
        
    if relative_addition is not None:
        resolution = (wmax-wmin) / len(wavelengths)
        plage_sum = int(relative_addition * sigma_exp / 2 / resolution) # plage sur laquelle on somme
    
    spectrum = np.zeros(len(wavelengths), dtype=float)

    def get_l_prob(v1, j1, v2, j2):
        _,_,_,E_1 = energy_function_C3P(v=v1, j=j1)
        _,_,_,E_2 = energy_function_B3P(v=v2, j=j2)
        Delta_E = E_1 - E_2 # cm-1
        l = 1/Delta_E * (1e7) # (nm)  longueur d'onde
        n_1 = population_distribution_B3P(v=v1, J=j1, T_el=T_el, T_rot=T_rot, T_vib=T_vib)/ (l*10**9)

        return l, get_transition_probability(l, n_1=n_1, L1=1, L2=1, v1=v1, v2=v2, J1=j1, J2=j2)
    
    def add_line(l, p):
        nonlocal spectrum
        if l < wmin or l > wmax:
            return
        if relative_addition is None:
            spectrum += shape(wavelengths, l, sigma_exp) * p * wavelengths
        else:
            index_middle = int(dichotomy_search(lambda x: wavelengths[int(x)]-l, 0, len(wavelengths)-1))
            for i in range(max(0, index_middle-plage_sum), min(len(wavelengths), index_middle+plage_sum)):
                spectrum[i] += shape(wavelengths[i], l, sigma_exp) * p * wavelengths[i]
                
    
    

    for v1 in range(0, nb_vib_levels):
        for v2 in range(0, nb_vib_levels):
            for i in range(0, nb_rot_levels):
                if i-1 >= 0:
                    j1, j2 = i, i-1 #! Raie P
                    l, p = get_l_prob(v1, j1, v2, j2)
                    add_line(l, p)

                    
                if i+1 < nb_rot_levels:
                    j1, j2 = i, i+1 #! Raie R   
                    l, p = get_l_prob(v1, j1, v2, j2)
                    add_line(l, p)

                if i > 0:   
                    j1, j2 = i, i
                    l, p = get_l_prob(v1, j1, v2, j2)#! Raie Q
                    add_line(l, p)
                    
    return spectrum/np.max(spectrum)