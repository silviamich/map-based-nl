import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from camb import model, initialpower
from matplotlib.pyplot import cm
from fgbuster.cosmology import _get_Cl_cmb
from functions import *
import os

# import cross-linking maps
a = hp.read_map("clinks_maps/full_a.fits")
b = hp.read_map("clinks_maps/full_b.fits")
c = hp.read_map("clinks_maps/full_c.fits")
d = hp.read_map("clinks_maps/full_d.fits")
e = hp.read_map("clinks_maps/full_e.fits")

f = hp.read_map("clinks_maps/full_f.fits")
g = hp.read_map("clinks_maps/full_g.fits")
h = hp.read_map("clinks_maps/full_h.fits")
j = hp.read_map("clinks_maps/full_j.fits")

mu = hp.read_map("clinks_maps/full_mu.fits")
nu = hp.read_map("clinks_maps/full_nu.fits")
gamma = hp.read_map("clinks_maps/full_gamma.fits")
delta = hp.read_map("clinks_maps/full_delta.fits")
eps = hp.read_map("clinks_maps/full_eps.fits")
lambd = hp.read_map("clinks_maps/full_lambda.fits")
kappa = hp.read_map("clinks_maps/full_kappa.fits")
eta = hp.read_map("clinks_maps/full_eta.fits")
zeta = hp.read_map("clinks_maps/full_zeta.fits")

detM = c*e - d**2 -a*(a*e-b*d) + b*(a*d-b*c)

# import sky maps
cmb_maps = []
noise_maps = []
bounds = []

Nsims = 100

for n in range(Nsims):
    cmb_maps.append(hp.read_map(f"/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/skymaps/cmb_{n}.fits", field=None))
    noise_maps.append(hp.read_map(f"/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/skymaps/noise_{n}.fits", field=None))
dipole_map = hp.read_map("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/skymaps/dipole.fits",field=None)
# combine the maps to build templates
I, Q, U = [], [], []

for n in range(Nsims):
    I.append((cmb_maps[n]+noise_maps[n]+dipole_map)[0])
    Q.append((cmb_maps[n]+noise_maps[n]+dipole_map)[1])
    U.append((cmb_maps[n]+noise_maps[n]+dipole_map)[2])

# these can be computed once for all:

averages_N_ell = build_noise_model(Nsims, noise_maps)

# build CMB spectra
nside = 64 
lmin, lmax = 2, 3*nside-1 
ell = np.arange(lmin,lmax)
r1 = 1e-3
Cl_bb_lens = _get_Cl_cmb(1.,0.)[2][lmin:lmax]
Cl_bb_prim = _get_Cl_cmb(0.,r1)[2][lmin:lmax]

r_grid = np.linspace(0,1e-3,num=1000) # grid of r to explore

##### --------- #####
sys = 'hwp'

G1 = np.logspace(-3,0,5) #valori realistici
A2F = np.linspace(1,5,5) #valori realistici

print(f'assessing sys {sys}')

if sys=='tes':
    for g_index in G1:
        # set systematics levels
        g1_one_over_k = []
        for n in range(Nsims):
            g1_one_over_k.append(-np.random.normal(loc=0, scale=g_index))

        # solve the binner equation
        d_nl, d_nl_c2, d_nl_s2 = [],[],[]
            
        for n in range(Nsims):
            d_nl.append(I[n] + a*Q[n] + b*U[n] + g1_one_over_k[n] * (I[n]**2 + c*Q[n]**2 + e*U[n]**2 +2*a*I[n]*Q[n] + 2*b*I[n]*U[n] + 2*d*Q[n]*U[n]))
            d_nl_c2.append(a*I[n] + c*Q[n] + d*U[n] + g1_one_over_k[n] * (a*I[n]**2 + f*Q[n]**2 + g*U[n]**2 + 2*c*I[n]*Q[n] + 2*d*I[n]*U[n] + 2*h*Q[n]*U[n]))
            d_nl_s2.append(b*I[n] + d*Q[n] + e*U[n] + g1_one_over_k[n] * (b*I[n]**2 + h*Q[n]**2 + j*U[n]**2 + 2*d*I[n]*Q[n] + 2*e*I[n]*U[n] + 2*g*Q[n]*U[n]))

        I_out = ((c*e-d**2)*d_nl + (b*d-a*e)*d_nl_c2 + (a*d-b*c)*d_nl_s2) / detM
        Q_out = ((b*d-a*e)*d_nl + (e-b**2)*d_nl_c2 + (a*b-d)*d_nl_s2) / detM
        U_out = ((a*d-b*c)*d_nl + (a*b-d)*d_nl_c2 + (c-a**2)*d_nl_s2) / detM

        assert not ((I_out == hp.UNSEEN) | np.isnan(I_out)).any(), "MAPS CONTAIN UNSEEN PIXELS!"
        #I_out[np.where(np.isnan(I_out))]=hp.UNSEEN
        #Q_out[np.where(np.isnan(Q_out))]=hp.UNSEEN
        #U_out[np.where(np.isnan(U_out))]=hp.UNSEEN
        output_maps =  np.stack((I_out, Q_out, U_out), axis=1) 
        output_maps = np.where(output_maps == hp.UNSEEN, hp.UNSEEN, output_maps * 1e6) #converted in uK!!!
        
        ##### save the maps
        os.makedirs("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/out_maps", exist_ok=True)
        for n in range(Nsims):
            file_path = os.path.join("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/out_maps", f"map_g{g_index}_{n}.fits")
            hp.write_map(file_path, output_maps[n], overwrite=True)
        #####
        
        #plot_spectra(ell, output_maps, Nsims, lmin, lmax, filename=f"spectra_{g_index}")

        #mean_likelihood = compute_and_plot_likelihoods(Nsims, output_maps, lmin, lmax, ell, averages_N_ell, r_grid, r1, filename=f"likelihoods_{g_index}",plot=False)    
        #bounds.append(find_bound(mean_likelihood, r_grid))

elif sys=='hwp':
    for a_index in A2F:
        amplitude_2f_k = []
        for n in range(Nsims):
            amplitude_2f_k.append(a_index)

        d_nl, d_nl_c2, d_nl_s2 = [],[],[]

        for n in range(Nsims):
            d_nl.append(I[n] + a*Q[n] + b*U[n] + amplitude_2f_k[n]*mu)
            d_nl_c2.append(a*I[n] + c*Q[n] + d*U[n] + amplitude_2f_k[n]*gamma)
            d_nl_s2.append(b*I[n] + d*Q[n] + e*U[n] + amplitude_2f_k[n]*eta)

        I_out = ((c*e-d**2)*d_nl + (b*d-a*e)*d_nl_c2 + (a*d-b*c)*d_nl_s2) / detM
        Q_out = ((b*d-a*e)*d_nl + (e-b**2)*d_nl_c2 + (a*b-d)*d_nl_s2) / detM
        U_out = ((a*d-b*c)*d_nl + (a*b-d)*d_nl_c2 + (c-a**2)*d_nl_s2) / detM

        assert not ((I_out == hp.UNSEEN) | np.isnan(I_out)).any(), "MAPS CONTAIN UNSEEN PIXELS!"
        #I_out[np.where(np.isnan(I_out))]=hp.UNSEEN
        #Q_out[np.where(np.isnan(Q_out))]=hp.UNSEEN
        #U_out[np.where(np.isnan(U_out))]=hp.UNSEEN
        output_maps =  np.stack((I_out, Q_out, U_out), axis=1) 
        output_maps = np.where(output_maps == hp.UNSEEN, hp.UNSEEN, output_maps * 1e6) #converted in uK!!!

        ##### save the maps
        os.makedirs("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/out_maps", exist_ok=True)
        for n in range(Nsims):
            file_path = os.path.join("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/out_maps", f"map_a{a_index}_{n}.fits")
            hp.write_map(file_path, output_maps[n], overwrite=True)
        #####


        #plot_spectra(ell, output_maps, Nsims, lmin, lmax, filename=f"spectra_{g_index}_{a_index}K")

        #mean_likelihood = compute_and_plot_likelihoods(Nsims, output_maps, lmin, lmax, ell, averages_N_ell, r_grid, r1, filename=f"likelihoods_{g_index}_{a_index}K",plot=False)    

        #bounds.append(find_bound(mean_likelihood, r_grid))

#print(bounds)
    
    
        
elif sys=='both':
    
    for g_index in G1:
        g1_one_over_k = []
        for a_index in A2F:
            amplitude_2f_k = []
            for n in range(Nsims):
                g1_one_over_k.append(-np.random.normal(loc=0, scale=g_index))
                amplitude_2f_k.append(a_index)

            d_nl, d_nl_c2, d_nl_s2 = [],[],[]

            for n in range(Nsims):
                d_nl.append(I[n] + a*Q[n] + b*U[n] + amplitude_2f_k[n]*mu + g1_one_over_k[n] * (I[n]**2 + c*Q[n]**2 + e*U[n]**2 +2*a*I[n]*Q[n] + 2*b*I[n]*U[n] + 2*d*Q[n]*U[n] + amplitude_2f_k[n]**2*nu + 2*amplitude_2f_k[n]*I[n]*mu + 2*amplitude_2f_k[n]*Q[n]*gamma + 2*amplitude_2f_k[n]*U[n]*eta))
                d_nl_c2.append(a*I[n] + c*Q[n] + d*U[n] + amplitude_2f_k[n]*gamma + g1_one_over_k[n] * (a*I[n]**2 + f*Q[n]**2 + g*U[n]**2 + 2*c*I[n]*Q[n] + 2*d*I[n]*U[n] + 2*h*Q[n]*U[n] + amplitude_2f_k[n]**2*delta + 2*amplitude_2f_k[n]*I[n]*gamma + 2*amplitude_2f_k[n]*Q[n]*eps + 2*amplitude_2f_k[n]*U[n]*zeta ))
                d_nl_s2.append(b*I[n] + d*Q[n] + e*U[n] + amplitude_2f_k[n]*eta + g1_one_over_k[n] * (b*I[n]**2 + h*Q[n]**2 + j*U[n]**2 + 2*d*I[n]*Q[n] + 2*e*I[n]*U[n] + 2*g*Q[n]*U[n] + amplitude_2f_k[n]**2*kappa + 2*amplitude_2f_k[n]*I[n]*eta + 2*amplitude_2f_k[n]*Q[n]*zeta + 2*amplitude_2f_k[n]*U[n]*lambd))

            I_out = ((c*e-d**2)*d_nl + (b*d-a*e)*d_nl_c2 + (a*d-b*c)*d_nl_s2) / detM
            Q_out = ((b*d-a*e)*d_nl + (e-b**2)*d_nl_c2 + (a*b-d)*d_nl_s2) / detM
            U_out = ((a*d-b*c)*d_nl + (a*b-d)*d_nl_c2 + (c-a**2)*d_nl_s2) / detM

            assert not ((I_out == hp.UNSEEN) | np.isnan(I_out)).any(), "MAPS CONTAIN UNSEEN PIXELS!"
            #I_out[np.where(np.isnan(I_out))]=hp.UNSEEN
            #Q_out[np.where(np.isnan(Q_out))]=hp.UNSEEN
            #U_out[np.where(np.isnan(U_out))]=hp.UNSEEN
            output_maps =  np.stack((I_out, Q_out, U_out), axis=1) 
            output_maps = np.where(output_maps == hp.UNSEEN, hp.UNSEEN, output_maps * 1e6) #converted in uK!!!

            ##### save the maps
            os.makedirs("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/out_maps", exist_ok=True)
            for n in range(Nsims):
                file_path = os.path.join("/Users/silviamicheli/Desktop/lb_sim/gen_telescope_scan/out_maps", f"map_g{g_index}_a{a_index}_{n}.fits")
                hp.write_map(file_path, output_maps[n], overwrite=True)
            #####


            #plot_spectra(ell, output_maps, Nsims, lmin, lmax, filename=f"spectra_{g_index}_{a_index}K")

            #mean_likelihood = compute_and_plot_likelihoods(Nsims, output_maps, lmin, lmax, ell, averages_N_ell, r_grid, r1, filename=f"likelihoods_{g_index}_{a_index}K",plot=False)    

            #bounds.append(find_bound(mean_likelihood, r_grid))

#print(bounds)







