import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xspectra.utils as utils
import xspectra.simulation as simulation
import xspectra.visualization as visualization


def process_spectrum(filename, output_folder, args, show_results=True):
    if args.verbose:
        print("Loading and preprocessing data...")
    data = np.loadtxt(filename, delimiter=args.delimiter, skiprows=args.length_header)
    data = utils.delete_background(data)

    if args.verbose:
        print("Calculating vibrational temperature...")
    i_primary, i_secondary = utils.find_index_primary_peaks(data[:, 0], data[:, 1], height=0.01, distance=100)
    T_vib = utils.calculate_temperature_single(data[:, 1], i1=i_primary, i2=i_secondary)

    if args.verbose:
        print("Calculating rotational temperature (R branch)...")
    T_rot_R_branch, T_rot_R_branch_err = utils.compute_Trot_with_branch(
        data[:, 0], data[:, 1], J_range=tuple(args.J_range), certainty=0.95
    )

    if args.verbose:
        print("Fitting simulation spectrum...")
    mask = (args.fit_limits[0] <= data[:, 0]) & (data[:, 0] <= args.fit_limits[1])
    filtered_wavelengths = data[mask, 0]
    filtered_spectrum = data[mask, 1]
    score, T_rot_sim, elargissement, decalage = utils.get_best_fit(
        filtered_wavelengths, filtered_spectrum, T_vib=T_vib, elargissement=0.1, w_decalage=0,
        T_range=tuple(args.T_range), elargissement_range=tuple(args.elargissement_range), 
        w_decalage_range=tuple(args.w_decalage_range), verbose=False, nb_steps=args.nb_steps
    )
    
    if args.verbose:
        print(f"Score of the fit: {score:.3f}")

    # Prepare results
    if args.verbose:
        print("Preparing results...")
    results = {
        "T_vib (K)": T_vib,
        "T_rot (K) - R branch": f"{T_rot_R_branch:.0f} ± {T_rot_R_branch_err:.0f}",
        "T_rot (K) - fit simulation": f"{T_rot_sim:.0f}",
        "Broadening (nm)": f"{elargissement:.3f}",
        "Wavelength Deviation (nm)": f"{decalage:.3f}",
    }
    results_df = pd.DataFrame([results])

    # Print results
    if show_results or args.verbose:
        print(results_df.to_markdown(index=False, tablefmt="grid"))

    # Save results if output folder is specified
    if output_folder:
        if args.verbose:
            print("Saving results and plots...")
        os.makedirs(output_folder, exist_ok=True)
        results_df.to_csv(os.path.join(output_folder, "results.csv"), index=False)

    # Save plots
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], label="Measured Spectrum")
    plt.plot(filtered_wavelengths, simulation.get_spectrum(filtered_wavelengths + decalage, T_el=1_000, T_vib=T_vib, T_rot=T_rot_sim, sigma_exp=elargissement), label="Fitted Simulation", linestyle="--")
    plt.axvline(x=args.fit_limits[0], color="r", linestyle="--", label="Fit Limits")
    plt.axvline(x=args.fit_limits[1], color="r", linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectrum Fit")
    plt.legend()
    plt.grid()
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, "spectrum_fit.png"), dpi=300, bbox_inches="tight")

    output_branch_method = os.path.join(output_folder, "Trot_calculation.png") if output_folder else "Trot_calculation.png"
    visualization.show_result_calculation_Trot(
        data[:, 0], data[:, 1], J_range=tuple(args.J_range), certainty=0.95, 
        output_file=output_branch_method, show=args.show_plots
    )

    # Show plots if requested
    if args.verbose:
        print("Process completed.")
    if args.show_plots:
        plt.show()
        
    return results_df

def main():
    parser = argparse.ArgumentParser(
        description=(
            "This script processes a spectrum file to calculate vibrational and rotational temperatures. "
            "It supports background removal, peak detection, and fitting with a simulation spectrum. "
            "Results can be saved to a specified folder, and plots can be displayed or saved."
        ),
        epilog=(
            "Examples:\n"
            "  # Process a single file with default parameters:\n"
            "  python -m xspectra.bin ./examples/data/spectrum.txt -o results\n\n"
            "  # Process all files in a directory:\n"
            "  python -m xspectra.bin ./examples/data/temperature_analysis -o res_bin\n\n"
            "  # Show plots and enable verbose output:\n"
            "  python -m xspectra.bin ./examples/data/spectrum.txt -v 1 -sp -o results\n\n"
            "  # Custom delimiter and header length:\n"
            "  python -m xspectra.bin spectrum.csv -d ',' -lh 2 -o results\n\n"
            "  # Custom fit limits and J range:\n"
            "  python -m xspectra.bin spectrum.txt -fl 335.0 338.5 -jr 10 25 -o results\n\n"
            "  # Advanced temperature and broadening settings:\n"
            "  python -m xspectra.bin spectrum.txt --T_range 200 1500 --elargissement_range 0.03 0.15\n\n"
            "  # Combine multiple parameters:\n"
            "  python -m xspectra.bin spectrum.txt -sp -v 1 -fl 335.0 338.5 -jr 10 25 --nb_steps 10\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("filename", type=str, help="Path to the spectrum file.")
    parser.add_argument("-o", "--output_folder", type=str, default=None, help="Folder to save results (optional).")
    parser.add_argument("-sp", "--show_plots", action="store_true", default=False, help="Show plots for fit and regression.")
    parser.add_argument("-d", "--delimiter", type=str, default="\t", help="Delimiter used in the spectrum file (default: '\t').")
    parser.add_argument("-lh", "--length_header", type=int, default=0, help="Number of header lines to skip in the spectrum file (default: 0).")
    parser.add_argument("-fl", "--fit_limits", type=float, nargs=2, default=(335.3, 338.0), help="Fit limits for the simulation as a tuple (lower, upper).")
    parser.add_argument("-jr", "--J_range", type=int, nargs=2, default=(8, 20), help="J range for rotational temperature calculation (default: 8 20).")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Set verbosity level (0: silent, 1: verbose).")
    parser.add_argument("--T_range", type=float, nargs=2, default=(100, 1200), help="Temperature range for the fit (default: 100 1200).")
    parser.add_argument("--elargissement_range", type=float, nargs=2, default=(0.05, 0.12), help="Broadening range for the fit (default: 0.05 0.12).")
    parser.add_argument("--w_decalage_range", type=float, nargs=2, default=(-2, 2), help="Wavelength deviation range for the fit (default: -2 2).")
    parser.add_argument("--nb_steps", type=int, default=5, help="Number of steps for the fit optimization (default: 5).")
    
    args = parser.parse_args()

    # Check if single file
    if os.path.isfile(args.filename):
        process_spectrum(args.filename, args.output_folder, args)
    elif os.path.isdir(args.filename):
        # Process all files in the directory with a progress bar

        results = pd.DataFrame()
        files = os.listdir(args.filename)
        for filename in tqdm(files, desc="Processing files"):
            output_folder = os.path.join(args.output_folder, os.path.splitext(filename)[0]) if args.output_folder else None
            file_path = os.path.join(args.filename, filename)
            res = process_spectrum(file_path, output_folder, args, show_results=False)
            res["Filename"] = filename  # Add filename column
            results = pd.concat([results, res], ignore_index=True)
        if args.output_folder:
            results.to_csv(os.path.join(args.output_folder, "results.csv"), index=False)
        print(results.to_markdown(index=False, tablefmt="grid"))
    else:
        print(f"Error: {args.filename} does not exist.")

    

if __name__ == "__main__":
    main()
