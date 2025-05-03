# X-Spectra

X-Spectra is a Python library designed for the spectral analysis of nitrogen plasma. It provides tools for data processing, spectral simulation, and visualization, making it easier to automate scientific calculations and analyze experimental data.

## Features

- **Data Processing**: Filter, normalize, and cut spectral data.
- **Spectral Analysis**: Extract peaks, calculate temperatures (vibrational and rotational), and analyze spectral lines.
- **Simulation**: Generate synthetic spectra using physical models.
- **Visualization**: Plot spectra, scores, and matches for better insights.

## Requirements

The following Python packages are required to use X-Spectra:

- `astropy==7.0.1`
- `astroquery==0.4.9.post1`
- `imageio==2.37.0`
- `matplotlib==3.10.1`
- `numpy==2.2.5`
- `pandas==2.2.3`
- `scipy==1.15.2`

You can install these dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/xspectra.git
cd xspectra
pip install -r requirements.txt
```

## Usage

Check the examples folder !

## Simulation 

The project includes a nitrogen spectrum simulation tool that, when adjusted to the studied spectrum, allows determining the rotational temperature, as well as the broadening and wavelength shift caused by calibration errors. This fitting process is performed using trichotomy, as the distance between the simulation spectrum and the studied spectrum is a convex function of the parameters, as shown in the following figure:

<div align="center">
    <img src="./pictures/scores_function.png" alt="scores_function" width="70%">
</div>

This results in robust fits, as demonstrated in the examples:

<div align="center">
    <img src="./pictures/spectrum_simulation.png" alt="fit" width="70%">
</div>

## Bin - Commmand line tool

The `bin` module provides a command-line interface for processing spectrum files to calculate vibrational and rotational temperatures. It supports background removal, peak detection, and fitting with a simulation spectrum. Results can be saved to a specified folder, and plots can be displayed or saved.

### Usage


```bash
>>> python -m xspectra.bin ./examples/data/temperature_analysis
Processing files: 100%|█████████████████████████████████████████| 3/3 [00:09<00:00,  3.28s/it]
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
|   T_vib (K) | T_rot (K) - R branch   |   T_rot (K) - fit simulation |   Broadening (nm) |   Wavelength Deviation (nm) | Filename   |
+=============+========================+==============================+===================+=============================+============+
|     808.233 | 302 ± 14               |                          255 |             0.086 |                       0.104 | 1.txt      |
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
|     823.692 | 312 ± 8                |                          269 |             0.086 |                       0.102 | 2.txt      |
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
|     832.07  | 363 ± 16               |                          305 |             0.085 |                       0.099 | 3.txt      |
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
```

Check the [documentation](docs.md) for more info !


## Documentation

Detailed documentation is available in the `docs.md` file. It includes descriptions and examples for all the functions provided by the library.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This library was developed as part of the MODAL project in 2025 for the spectral study of nitrogen plasma.

---

# X-Spectra (Français)

X-Spectra est une bibliothèque Python conçue pour l'analyse spectrale du plasma d'azote. Elle fournit des outils pour le traitement des données, la simulation spectrale et la visualisation, facilitant ainsi les calculs scientifiques automatisés et l'analyse des données expérimentales.

## Fonctionnalités

- **Traitement des données** : Filtrer, normaliser et découper les données spectrales.
- **Analyse spectrale** : Extraire les pics, calculer les températures (vibrationnelle et rotationnelle) et analyser les raies spectrales.
- **Simulation** : Générer des spectres synthétiques à l'aide de modèles physiques.
- **Visualisation** : Tracer des spectres, des scores et des correspondances pour une meilleure compréhension.

## Prérequis

Les packages Python suivants sont nécessaires pour utiliser X-Spectra :

- `astropy==7.0.1`
- `astroquery==0.4.9.post1`
- `imageio==2.37.0`
- `matplotlib==3.10.1`
- `numpy==2.2.5`
- `pandas==2.2.3`
- `scipy==1.15.2`

Vous pouvez installer ces dépendances en utilisant le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

## Installation

Clonez le dépôt et installez les dépendances requises :

```bash
git clone https://github.com/your-repo/xspectra.git
cd xspectra
pip install -r requirements.txt
```

## Utilisation

Consultez le dossier `examples` !

## Simulation

Le projet contient un outil de simulation de spectre de diazote qui permet en l'ajustant au spectre étudié de trouver la température rotationnelle, ainsi que l'élargissement et le décalage en longueur d'onde dû aux mauvaises calibrations. Ce *fit* sur les données est effectués par trichotomie, la distance du spectre de simulation au spectre étudié étant une fonction convexe des paramètres, comme on peut le voir sur la figure suivante : 

<div align="center">
    <img src="./pictures/scores_function.png" alt="scores_function" width="70%">
</div>

Ce qui nous donnes des fits robustes comme on peut le voir dans les examples : 

<div align="center">
    <img src="./pictures/spectrum_simulation.png" alt="fit" width="70%">
</div>

## Bin - Outil en ligne de commande

Le module `bin` fournit une interface en ligne de commande pour traiter les fichiers de spectres afin de calculer les températures vibrationnelle et rotationnelle. Il prend en charge la suppression du bruit de fond, la détection des pics et l'ajustement avec un spectre simulé. Les résultats peuvent être sauvegardés dans un dossier spécifié, et les graphiques peuvent être affichés ou sauvegardés.

### Exemple d'utilisation

```bash
>>> python -m xspectra.bin ./examples/data/temperature_analysis
Traitement des fichiers : 100%|█████████████████████████████████████████| 3/3 [00:09<00:00,  3.28s/it]
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
|   T_vib (K) | T_rot (K) - R branch   |   T_rot (K) - fit simulation |   Broadening (nm) |   Wavelength Deviation (nm) | Filename   |
+=============+========================+==============================+===================+=============================+============+
|     808.233 | 302 ± 14               |                          255 |             0.086 |                       0.104 | 1.txt      |
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
|     823.692 | 312 ± 8                |                          269 |             0.086 |                       0.102 | 2.txt      |
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
|     832.07  | 363 ± 16               |                          305 |             0.085 |                       0.099 | 3.txt      |
+-------------+------------------------+------------------------------+-------------------+-----------------------------+------------+
```

Consultez la [documentation](docs.md) pour plus d'informations !

## Documentation

Une documentation détaillée est disponible dans le fichier `docs.md`. Elle inclut des descriptions et des exemples pour toutes les fonctions fournies par la bibliothèque.

## Contribuer

Les contributions sont les bienvenues ! Si vous avez des suggestions ou des améliorations, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus de détails.

## Remerciements

Cette bibliothèque a été développée dans le cadre du projet MODAL en 2025 pour l'étude spectrale du plasma d'azote.

