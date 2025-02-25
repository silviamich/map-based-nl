import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from camb import model, initialpower
from matplotlib.pyplot import cm
from fgbuster.cosmology import _get_Cl_cmb

def plot_spectra(ell, spectrum, Nsims, lmin, lmax, filename=str):
    """
    Plot TT, EE, and BB power spectra using subplots.
    
    Parameters:
        ell (array): Multipole moments.
        spectrum (list or array): Simulated maps.
        Nsims (int): Number of simulations.
        lmin (int): Minimum multipole for plotting.
        lmax (int): Maximum multipole for plotting.
        save (bool): Whether to save the figure automatically.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    colors = ['k', 'g', 'red']
    labels = [r'$D_{\ell}^{TT} [\mu K^2]$', r'$D_{\ell}^{EE} [\mu K^2]$', r'$D_{\ell}^{BB} [\mu K^2]$']
    
    for i, ax in enumerate(axes):
        for n in range(Nsims):
            cl = hp.anafast(spectrum[n])[i, lmin:lmax]  # Compute power spectrum
            ax.loglog(ell, ell * (ell + 1) * cl / (2 * np.pi), color=colors[i], alpha=0.2)

        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(labels[i])
        ax.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.tight_layout()  
    
    plt.savefig(filename+".png", dpi=300, bbox_inches='tight')


    
def build_noise_model(Nsims, noise_maps):
    N_ell = []
    for n in range(Nsims):
        N_ell.append(hp.anafast(1e6*noise_maps[n])[2]) # converted to uK

    N_ell = np.array(N_ell)
    # Compute leave-one-out means to avoid correlations in the noise
    averages_N_ell = np.array([(np.sum(N_ell, axis=0) - N_ell[n]) / (Nsims-1) for n in range(Nsims)])
    
    return averages_N_ell


def compute_and_plot_likelihoods(Nsims, maps, lmin, lmax, ell, noise_model, r_grid, r1, filename=str, plot=True):
    """
    Computes and plots the likelihood distribution over `r_grid` for multiple simulations.

    Parameters:
        Nsims (int): Number of simulations.
        maps (array): Simulated CMB maps.
        lmin (int): Minimum multipole for likelihood evaluation.
        lmax (int): Maximum multipole for likelihood evaluation.
        ell (array): Array of multipole moments.
        noise_model (array): Noise power spectrum.
        r_grid (array): Array of r values to evaluate likelihood over.
        r1 (float): Reference tensor-to-scalar ratio.
    """
    
    def exact_likelihood_func(r, Cl_BB_obs, Cl_BB_prim, Cl_lens, Nl):
        """Computes log-likelihood and normalized likelihood for a given r."""
        Cl_BB_model = Cl_BB_prim * (r / r1) + Cl_lens + Nl
        logL = np.sum((2 * ell + 1) / 2 * (Cl_BB_obs / Cl_BB_model - np.log(Cl_BB_obs / Cl_BB_model) - 1))
        return logL

    # Initialize storage for likelihoods
    mean_likelihood = []

    if plot==True: plt.figure(figsize=(15, 4))

    for i in range(Nsims):   
        Cl_BB_obs = hp.anafast(maps[i])[2, lmin:lmax]  # Compute observed B-mode power spectrum
        Cl_BB_prim = _get_Cl_cmb(0., r1)[2][lmin:lmax]  # Primordial B-modes
        Cl_lens = _get_Cl_cmb(1., 0.)[2][lmin:lmax]  # Lensed B-modes
        Nl = noise_model[i][lmin:lmax]  # Noise power spectrum

        # Compute log-likelihood over r_grid
        logL = np.array([exact_likelihood_func(r, Cl_BB_obs, Cl_BB_prim, Cl_lens, Nl) for r in r_grid])

        # Normalize likelihood
        likelihood = np.exp(-logL)
        likelihood /= np.max(likelihood)  # Normalize to 1

        mean_likelihood.append(logL)

        if plot==True: plt.plot(r_grid, likelihood, alpha=0.5, color='k', linewidth=0.2)

    # Compute mean likelihood across simulations
    mean_likelihood = np.array(mean_likelihood)
    mean_likelihood = np.exp(-np.mean(mean_likelihood, axis=0))
    mean_likelihood /= np.max(mean_likelihood)

    if plot==True:
    
        plt.plot(r_grid, mean_likelihood, color='r', linewidth=1)

        plt.xlabel(r'$r$')
        plt.ylabel("Normalized Likelihood")
    
        # Save the figure
        plt.savefig(filename+".png", dpi=300, bbox_inches='tight')
    
    return mean_likelihood
    

def find_bound(likelihood, r_array, level=0.68):
    """Finds the r values where the cumulative likelihood crosses the given confidence level."""
    cumulative = np.cumsum(likelihood) / np.sum(likelihood)    
    crossing_indices = np.where(np.diff(np.sign(cumulative - level)))[0]
    return r_array[crossing_indices]
