import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
import scienceplots
from matplotlib.backends.backend_pgf import FigureCanvasPgf

matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
plt.style.use(['science','grid'])
plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["axes.formatter.limits"] = [-2, 2]

def gauss(x: np.ndarray, sigma: float, a: float) -> np.ndarray:
    return (
        a/np.sqrt(np.pi*(sigma**2))
    )*np.exp(
        -1 * (x**2/(sigma**2))
    )

def plot_atom_distribution(df):
    """
    Plots the distribution of atoms in a DataFrame.
    
    :param df: DataFrame containing atom data with 'atom' and 'count' columns.
    """
    deltas = []
    fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(15, 10), constrained_layout=True)
    for idx, measure_time in enumerate(df.iloc[:, 1:]):
        dist = df['bin'].values*1e-6  # Convert bin to meters
        axs.flatten()[idx].plot(dist, df[measure_time], label="Atom position")
        params, cov = curve_fit(gauss, dist, df[measure_time], p0=[1e-6, 1])
        deltas.append(params[0])
        locs = np.linspace(dist.min(), dist.max(), 1000)
        fit_Y = gauss(locs, *params)
        axs.flatten()[idx].plot(locs, fit_Y, label="Gaussian fit", linestyle='--')
        axs.flatten()[idx].set_title(f"Measure Time: {measure_time[1:-3]} $\mu$s")
    for ax in axs.flat:
        ax.legend()
        ax.grid(True)
    fig.supxlabel('Atom Position')
    fig.supylabel('Atom Count')
    plt.suptitle('Atom Distribution Over Time')
    plt.show()
    return deltas
    

def read_csv(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data from the CSV file.
    """
    return pandas.read_csv(file_path)

def main():
    # Read the CSV file
    df = read_csv("atom_distributions.csv")
    deltas = plot_atom_distribution(df)
    lingres = stats.linregress(np.arange(len(deltas))*1e-6, deltas)
    print(f"Linear regression results: slope={lingres.slope:.2e}, intercept={lingres.intercept:.2e}, r_value={lingres.rvalue:.2e}, p_value={lingres.pvalue:.2e}, std_err={lingres.stderr:.2e}")
    mass = sp.constants.hbar / (0.07e-6 * lingres.slope)
    print(f"Mass of the atom: {mass:.2e} kg or {mass * 6.022e26:.2f} amu")

if __name__ == "__main__":
    main()