#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:15:21 2021

@author: pierre
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate


# K band
F_0 = 4.283e-14*1e4# *1e4 pour passer de W/cm2/um à W/m2/um
bwidth = 0.261
filter_name = '/home/pierre/Documents/2021/SINFONI_analysis/2MASS_filters/K.txt'
filt = np.loadtxt(filter_name).T


# H+K

# NGC 1068 0.025
mag = 8.9
rad_pix = 21
coadd_name_025 = '/media/pierre/Disque_2/SNR/SORTED_COADD/NGC1068/H+K/0.025/NGC1068-25mas-HK-1_COADD_OBJ_53653.21416709.fits'
# coadd_2_name = '/media/pierre/Disque_2/SNR/SORTED_COADD/NGC1068/H+K/0.025/NGC1068-25mas-HK-1_COADD_OBJ_53653.24721669.fits'
# coadd_3_name = '/media/pierre/Disque_2/SNR/SORTED_COADD/NGC1068/H+K/0.025/NGC1068-25mas-HK-1_COADD_OBJ_54067.13634114.fits'
# coadd_4_name = '/media/pierre/Disque_2/SNR/SORTED_COADD/NGC1068/H+K/0.025/NGC1068-25mas-HK-1_COADD_OBJ_54067.15073658.fits'

spec_star_filename = '/media/pierre/Disque_2/SNR/archive/076B0098A/reflex_end_products/2021-04-23T11:22:26/SINFO.2005-10-10T08:49:48.550_tpl/Telluric_Standard_STD_STAR_SPECTRA.fits'
typ = 'g2v'
savename='/home/pierre/Documents/2021/SINFONI_analysis/SINFONI_CALIB/NGC_1068/K_25mas.fits'
# pixs = xmax-28:xmax+28,ymax-28:ymax+28

# # NGC 1068 0.1
# mag = 8.9
# rad_pix=5
# coadd_name_100 = '/media/pierre/Disque_2/SNR/SORTED_COADD/NGC1068/H+K/0.1/NGC1068-100mas-HK-1_COADD_OBJ_53649.24330859.fits'

# spec_star_filename = '/media/pierre/Disque_2/SNR/archive/076B0098A/reflex_end_products/2021-04-23T11:22:26/SINFO.2005-10-06T07:06:02.984_tpl/Telluric_Standard_STD_STAR_SPECTRA.fits'
# typ = 'g2v'
# savename='/home/pierre/Documents/2021/SINFONI_analysis/SINFONI_CALIB/NGC_1068/K_100mas.fits'
# # pixs = xmax-28:xmax+28,ymax-28:ymax+28


def get_obj(filename):
    hdu = fits.open(filename)
    obj = hdu[0].data
    return np.nan_to_num(obj)
    

def get_info(filename):
    hdu = fits.open(filename)
    prog_id = hdu[0].header['HIERARCH ESO OBS PROG ID'].replace('.','').replace('-','').replace('(','').replace(')','')
    filt = hdu[0].header['HIERARCH ESO INS GRAT1 NAME'].replace(' ','')
    pix_scale = hdu[0].header['HIERARCH ESO INS OPTI1 NAME'].replace(' ','')
    date = hdu[0].header['MJD-OBS']
    return prog_id, date, filt, pix_scale

def get_obj_info(filename):
    hdu = fits.open(filename)
    prog_id = hdu[0].header['HIERARCH ESO OBS PROG ID'].replace('.','').replace('-','').replace('(','').replace(')','')
    obj_name = hdu[0].header['HIERARCH ESO OBS TARG NAME']
    date = hdu[0].header['MJD-OBS']
    return obj_name, date


def find_std_stars(filename, rep='/media/pierre/Disque_2/SNR/archive'):
    prog_id, date, filt, pix_scale = get_info(filename)
    rep_of_interest = rep+'/'+prog_id+'/reflex_end_products'
    for rep in glob.glob(rep_of_interest+'/*'):
        print(rep)
        for re in glob.glob(rep+'/*'):
            star_spectra = glob.glob(re+'/*STD_STAR_SPECTRA.fits')
            for spec in star_spectra:
                std_id, date, std_filt, std_pix = get_info(spec)
                if std_id == prog_id and std_filt == filt and std_pix == pix_scale:
                    print(star_spectra, get_obj_info(spec), get_info(spec))

def load_th_spec(typ, rep='/home/pierre/Documents/2021/SINFONI_analysis/pickles_atlas/'):
    full = np.loadtxt(rep+'/uk'+typ+'.dat')
    return full[:,0], full[:,1], full[:,2]

#%%

# find_std_stars(coadd_1_name)


spec_star_wl = fits.open(spec_star_filename)[1].data['wavelength']
spec_star_spectrum = fits.open(spec_star_filename)[1].data['counts_bkg']
spec_star_bg = fits.open(spec_star_filename)[1].data['bkg_tot']
spec_star_tot = fits.open(spec_star_filename)[1].data['counts_tot']


w, f, e = load_th_spec(typ)
spec_th = np.interp(spec_star_wl, w/1e4, f)

transmi = np.nan_to_num(spec_star_spectrum/spec_th)
transmi /= np.median(transmi)
transmi[transmi<5e-2]=5e-2


obj = get_obj(coadd_name_025)
# obj = get_obj(coadd_2_name)
# obj3 = get_obj(coadd_3_name)
# obj4 = get_obj(coadd_4_name)
# obj = np.mean([obj1,obj2,obj3,obj4],0)
ymax, xmax = np.unravel_index(np.argmax(np.median(np.nan_to_num(obj),0)), np.shape(obj[0]))
obj_detransmitted = obj/transmi[:, np.newaxis, np.newaxis]

filter_interp = np.interp(spec_star_wl, filt[0], filt[1])
obj_2MASSed = obj_detransmitted*filter_interp[:, np.newaxis, np.newaxis]

x = np.arange(len(obj_2MASSed[0,0]))
y = np.arange(len(obj_2MASSed[0]))
xx, yy = np.meshgrid(x,y)



mask = ((xx-xmax)**2+(yy-ymax)**2)**0.5<rad_pix
# tot_adu = np.sum(np.nan_to_num(obj_2MASSed[:,xmax-28:xmax+28,ymax-28:ymax+28]))
# tot_adu = np.sum(np.nan_to_num(obj_2MASSed[:,xmax-5:xmax+5,ymax-5:ymax+5]))
tot_adu = np.sum(np.sum(obj_2MASSed,0)*mask)
tot_flux = F_0*10**(-0.4*mag)*bwidth
adu = tot_flux/tot_adu
dwl = spec_star_wl[2]-spec_star_wl[1]
obj_final = obj_detransmitted*adu

# hdu = fits.open(coadd_name)
# hdu[0].data = obj_final

# hdu.writeto(savename)

#%%
import sys
sys.path.append('/home/pierre/Documents/2021/NEW_SPHERE')

from spec_obj import Spec_Obj
from copy import deepcopy
from scipy.ndimage import shift, rotate


pos = np.loadtxt('/home/pierre/Documents/2021/NEW_SPHERE/REDUCED_DATA/pos')
lam = np.loadtxt('/home/pierre/Documents/2021/NEW_SPHERE/REDUCED_DATA/lam')
obj = np.loadtxt('/home/pierre/Documents/2021/NEW_SPHERE/REDUCED_DATA/obj')
obj_std = np.loadtxt('/home/pierre/Documents/2021/NEW_SPHERE/REDUCED_DATA/obj_std')
cal = np.loadtxt('/home/pierre/Documents/2021/NEW_SPHERE/REDUCED_DATA/cal')
cal_std = np.loadtxt('/home/pierre/Documents/2021/NEW_SPHERE/REDUCED_DATA/cal_std')

sphobj = Spec_Obj(obj, obj_std, lam, pos)

sinforate = rotate(obj_final, 12, axes=(2, 1), reshape=False,order=3)
ymax, xmax = np.unravel_index(np.argmax(np.median(np.nan_to_num(sinforate),0)), np.shape(sinforate[0]))

obj_for_slit = deepcopy(sinforate)
for k in range(len(obj_for_slit)):
    im = sinforate[k]
    # ym, xm = np.unravel_index(np.argmax(im), np.shape(im))
    # print(xm, ym)
    # shifted_im = shift(im, (ymax-ym,xmax-xm+0.5), order=3)
    # obj_for_slit[k] = shifted_im
    obj_for_slit[k] = im
    
slit = np.sum(sinforate[90:2078,:,34:38],2)*0.9


slit = np.sum(obj_for_slit[90:2078,:,48:50],2)*90/100
err = np.ones(np.shape(slit))*0.05*slit

xm = np.argmax(np.mean(slit,0))-1
pos = (np.arange(np.shape(slit)[1])-xm)*0.0125
pos = (np.arange(np.shape(slit)[1])-xm)*0.05
wl = spec_star_wl[90:2078]*1e3

sinfobj = Spec_Obj(slit, err, wl, pos)


plt.figure()
sinfobj.plot_spec(0.25,0.35)
lsinf, nod_sinf_1, nod_sinf_1_err = sinfobj.make_spec(0.25,0.35)
sphobj.plot_spec(0.25,0.35)
lsph, nod_sph_1, nod_sph_1_err = sphobj.make_spec(0.25,0.35)

lj = lsph[:np.argmin((lsinf[0]-lsph)**2)]
jnod1 = nod_sph_1[:np.argmin((lsinf[0]-lsph)**2)]
jerr1 = nod_sph_1_err[:np.argmin((lsinf[0]-lsph)**2)]

lh = lsph[np.argmin((lsinf[0]-lsph)**2)+1:]
lh_sinf = lsinf[:np.argmin((lsinf-lsph[-1])**2)]
hnod1_sph = nod_sph_1[np.argmin((lsinf[0]-lsph)**2)+1:]
hnod1_sinf = nod_sinf_1[:np.argmin((lsinf-lsph[-1])**2)]
hnod1_sinfinterp = np.interp(lh, lh_sinf, hnod1_sinf)
hnod1 = np.mean([hnod1_sinfinterp, hnod1_sph],0)
herr1 = np.std([hnod1_sinfinterp, hnod1_sph*np.median(hnod1_sinfinterp)/np.median(hnod1_sph)],0)


lk = lsinf[np.argmin((lsinf-lsph[-1])**2):]
knod1 = nod_sinf_1[np.argmin((lsinf-lsph[-1])**2):]
kerr1 = nod_sinf_1_err[np.argmin((lsinf-lsph[-1])**2):]/9


plt.figure()
sinfobj.plot_spec(0.65,0.75)
lsinf, nod_sinf_2, nod_sinf_2_err = sinfobj.make_spec(0.65,0.75)
sphobj.plot_spec(0.65,0.75)
lsph, nod_sph_2, nod_sph_2_err = sphobj.make_spec(0.65,0.75)

lj = lsph[:np.argmin((lsinf[0]-lsph)**2)]
jnod2 = nod_sph_2[:np.argmin((lsinf[0]-lsph)**2)]
jerr2 = nod_sph_2_err[:np.argmin((lsinf[0]-lsph)**2)]

lh = lsph[np.argmin((lsinf[0]-lsph)**2)+1:]
lh_sinf = lsinf[:np.argmin((lsinf-lsph[-1])**2)]
hnod2_sph = nod_sph_2[np.argmin((lsinf[0]-lsph)**2)+1:]
hnod2_sinf = nod_sinf_2[:np.argmin((lsinf-lsph[-1])**2)]
hnod2_sinfinterp = np.interp(lh, lh_sinf, hnod2_sinf)
hnod2 = np.mean([hnod2_sinfinterp, hnod2_sph],0)
herr2 = np.std([hnod2_sinfinterp, hnod2_sph*np.median(hnod2_sinfinterp)/np.median(hnod2_sph)],0)


lk = lsinf[np.argmin((lsinf-lsph[-1])**2):]
knod2 = nod_sinf_2[np.argmin((lsinf-lsph[-1])**2):]
kerr2 = nod_sinf_2_err[np.argmin((lsinf-lsph[-1])**2):]/3


plt.figure()
sinfobj.plot_spec(-0.035,0.065)
lsinf, nod_sinf_0, nod_sinf_0_err = sinfobj.make_spec(-0.035,0.065)
nod_sinf_0 *= 2.4/0.52
sphobj.plot_spec(-0.035,0.065)
lsph, nod_sph_0, nod_sph_0_err = sphobj.make_spec(-0.035,0.065)

lj = lsph[:np.argmin((lsinf[0]-lsph)**2)]
jnod0 = nod_sph_0[:np.argmin((lsinf[0]-lsph)**2)]
jerr0 = nod_sph_0_err[:np.argmin((lsinf[0]-lsph)**2)]

lh = lsph[np.argmin((lsinf[0]-lsph)**2)+1:]
lh_sinf = lsinf[:np.argmin((lsinf-lsph[-1])**2)]
hnod0_sph = nod_sph_0[np.argmin((lsinf[0]-lsph)**2)+1:]
hnod0_sinf = nod_sinf_0[:np.argmin((lsinf-lsph[-1])**2)]
hnod0_sinfinterp = np.interp(lh, lh_sinf, hnod0_sinf)
hnod0 = np.mean([hnod0_sinfinterp, hnod0_sph],0)
herr0 = np.std([hnod0_sinfinterp, hnod0_sph*np.median(hnod0_sinfinterp)/np.median(hnod0_sph)],0)


lk = lsinf[np.argmin((lsinf-lsph[-1])**2):]
knod0 = nod_sinf_0[np.argmin((lsinf-lsph[-1])**2):]
kerr0 = nod_sinf_0_err[np.argmin((lsinf-lsph[-1])**2):]/3

plt.figure()
plt.errorbar(lk, knod1, kerr1)
plt.errorbar(lh, hnod1, herr1)
plt.errorbar(lj, jnod1, jerr1)

plt.figure()
plt.errorbar(lk, knod2, kerr2)
plt.errorbar(lh, hnod2, herr2)
plt.errorbar(lj, jnod2, jerr2)

plt.figure()
plt.errorbar(lk, knod0, kerr0)
plt.errorbar(lh, hnod0, herr0)
plt.errorbar(lj, jnod0, jerr0)

surface = 0.09*0.1

# np.savetxt('./PRODUITS/lambdas_k.dat', lk)
# np.savetxt('./PRODUITS/lambdas_h.dat', lh)
# np.savetxt('./PRODUITS/lambdas_j.dat', lj)

# np.savetxt('./PRODUITS/jnod0.dat', jnod0)
# np.savetxt('./PRODUITS/hnod0.dat', hnod0)
# np.savetxt('./PRODUITS/knod0.dat', knod0)

# np.savetxt('./PRODUITS/jnod1.dat', jnod1)
# np.savetxt('./PRODUITS/hnod1.dat', hnod1)
# np.savetxt('./PRODUITS/knod1.dat', knod1)

# np.savetxt('./PRODUITS/jnod2.dat', jnod2)
# np.savetxt('./PRODUITS/hnod2.dat', hnod2)
# np.savetxt('./PRODUITS/knod2.dat', knod2)

# np.savetxt('./PRODUITS/jerr0.dat', jerr0)
# np.savetxt('./PRODUITS/herr0.dat', herr0)
# np.savetxt('./PRODUITS/kerr0.dat', kerr0)

# np.savetxt('./PRODUITS/jerr1.dat', jerr1)
# np.savetxt('./PRODUITS/herr1.dat', herr1)
# np.savetxt('./PRODUITS/kerr1.dat', kerr1)

# np.savetxt('./PRODUITS/jnod2.dat', jnod2)
# np.savetxt('./PRODUITS/hnod2.dat', hnod2)
# np.savetxt('./PRODUITS/knod2.dat', knod2)

# np.savetxt('./PRODUITS/surface.dat', [surface])

# a, b1, c = sinfobj.make_spec(0.65,0.75)
# a, b2, c = sinfobj.make_spec(0.55,0.65)
# a, b3, c = sinfobj.make_spec(0.75,0.95)

# ad, b4, c = ngc1068.make_spec(0.65,0.75)
# ad, b5, c = ngc1068.make_spec(0.55,0.65)
# ad, b6, c = ngc1068.make_spec(0.75,0.95)

# plt.plot(a, b1-(b2+b3/2))
# plt.plot(ad, b4-(b5+b6/2))


# a, b1, c = sinfobj.make_spec(0.25,0.35)
# a, b2, c = sinfobj.make_spec(0.35,0.45)
# a, b3, c = sinfobj.make_spec(0.18,0.25)

# ad, b4, c = ngc1068.make_spec(0.25,0.35)
# ad, b5, c = ngc1068.make_spec(0.35,0.45)
# ad, b6, c = ngc1068.make_spec(0.15,0.25)

# plt.plot(a, b1-(b2+b3)/2)
# plt.plot(ad, b4-(b5+b6)/2)

# sinfobj.plot_spec(-0.2,-0.1)
# ngc1068.plot_spec(-0.2,-0.1)

# plt.figure()
# sinfobj.plot_spec(-.5,.5)
# ngc1068.plot_spec(-0.5,0.5)


# import pickle 
# naco = pickle.load(open('/media/pierre/Disque_2/ORDI_1/2018/NACO/REDUCTION_2/PRODUITS/NOT_CAL/K_-103-3/objet.p', 'rb'))
# naco = pickle.load(open('/media/pierre/Disque_2/ORDI_1/2018/NACO/REDUCTION_2/PRODUITS/NOT_CAL/L_12/objet.p', 'rb'))
