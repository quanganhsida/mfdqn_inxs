# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from ismember import ismember

def create_layout(deploy_param):
    N = deploy_param.num_of_subnetworks
    bound = deploy_param.deploy_length - 2*deploy_param.subnet_radius
    X = np.zeros([deploy_param.num_of_subnetworks,1],dtype=np.float64)
    Y = np.zeros([deploy_param.num_of_subnetworks,1],dtype=np.float64)
    dist_2 = deploy_param.minDistance**2
    loop_terminate = 1
    nValid = 0
    while nValid < deploy_param.num_of_subnetworks and loop_terminate < 1e6:
        newX = bound*(deploy_param.rng_value.uniform()-0.5)
        newY = bound*(deploy_param.rng_value.uniform()-0.5)
        if all(np.greater(((X[0:nValid] - newX)**2 + (Y[0:nValid] - newY)**2),dist_2)):
            X[nValid] = newX
            Y[nValid] = newY
            nValid = nValid+1
        loop_terminate = loop_terminate+1
    if nValid < deploy_param.num_of_subnetworks:
        print("Invalid number of subnetworks for deploy size")
        exit()

    #Location of the access points
    X = X+deploy_param.deploy_length/2
    Y = Y+deploy_param.deploy_length/2
    gwLoc = np.concatenate((X, Y), axis=1)
    dist_rand = deploy_param.rng_value.uniform(low=deploy_param.minD, high=deploy_param.subnet_radius, size=[N,1])
    angN = deploy_param.rng_value.uniform(low=0, high=2*np.pi, size=[N,1])
    D_XLoc = X + dist_rand*np.cos(angN)
    D_YLoc = Y + dist_rand*np.sin(angN)
    dvLoc = np.concatenate((D_XLoc, D_YLoc), axis=1)
    dist = cdist(gwLoc,dvLoc)
    return dist, gwLoc, dvLoc

def createMap(self):
    N1 = len(self.mapXPoints)
    N2 = len(self.mapYPoints)
    if 'Gamma' not in globals(): #if Gamma is already defined do not recalculate
        G = np.zeros([N1,N2],dtype=np.float64)
        for n in range(N1):
            for m in range(N2):
                G[n,m]= self.shadStd*np.exp(-1*np.sqrt(np.min([np.absolute(self.mapXPoints[0]-self.mapXPoints[n]),\
                                          np.max(self.mapXPoints)-np.absolute(self.mapXPoints[0]-self.mapXPoints[n])])**2\
                + np.min([np.absolute(self.mapYPoints[0]-self.mapYPoints[m]),np.max(self.mapYPoints)\
                      -np.absolute(self.mapYPoints[0]-self.mapYPoints[m])])**2)/self.correlationDistance)
        global Gamma
        Gamma = np.fft.fft2(G)
    Z = np.random.randn(N1,N2) + 1j*np.random.randn(N1,N2)
    mapp = np.real(np.fft.fft2(np.multiply(np.sqrt(Gamma),Z)\
                               /np.sqrt(N1*N2)))*np.sqrt(2)
    self.mapp = mapp
    return mapp

def channel_pathLoss(deploy_param, dist):
    PrLoS = np.exp(dist * np.log(1 - deploy_param.clutDens) / deploy_param.clutSize)
    NLoS = PrLoS <= (1 - PrLoS)
    Gamma = 31.84 + 21.5 * np.log10(dist) + 19 * np.log10(deploy_param.fc/1e9) #[idx]
    if deploy_param.clutType == 'sparse':
        Gamma[NLoS] = np.max([Gamma[NLoS],
                            33 + 25.5 * np.log10(dist[NLoS]) + 20 * np.log10(deploy_param.fc/1e9)],
                            axis=0)
    elif deploy_param.clutType == 'dense':
        Gamma[NLoS] = np.max([Gamma[NLoS],
                            33 + 25.5 * np.log10(dist[NLoS]) + 20 * np.log10(deploy_param.fc/1e9),
                            18.6 + 35.7 * np.log10(dist[NLoS]) + 20 * np.log10(deploy_param.fc/1e9)],
                            axis=0)
    return 10**(-Gamma/10)

def computeShadowing(deploy_param, dist, gwLoc):
    Ilocx,idx = ismember(np.round(gwLoc[0, :], decimals=1), np.round(deploy_param.mapXPoints, decimals=1))
    Ilocy,idy = ismember(np.round(gwLoc[1, :], decimals=1), np.round(deploy_param.mapYPoints, decimals=1))
    mapp = createMap(deploy_param)
    idxx = np.ravel_multi_index([idx, idy], (mapp.shape[0], mapp.shape[0]))
    mapp1 = mapp.flatten()
    f = mapp1[idxx]
    f = f.reshape(1, -1)
    fAB = np.add(f.T, f)
    S = np.multiply(np.divide((1 - np.exp(-1 * dist / deploy_param.correlationDistance)), \
                              (np.sqrt(2) * np.sqrt((1 \
                                                     + np.exp(-1 * dist / deploy_param.correlationDistance))))), fAB)
    return S, mapp

def compute_power(deploy_param, dist, Loc, K):
    N = deploy_param.num_of_subnetworks
    power = np.zeros((K, N, N))
    S, mapp = computeShadowing(deploy_param, dist, Loc.T)
    S_linear = 10**(-S/10)
    PL = channel_pathLoss(deploy_param, dist)
    for k in range(K):
        h = (1 / np.sqrt(2)) * (deploy_param.rng_value.randn(N, N) + 1j * deploy_param.rng_value.randn(N, N))
        power[k, :, :] = PL * S_linear * np.power(np.abs(h), 2)
    return power, PL

def generate_samples(deploy_param, number_of_snapshots):
    N = deploy_param.num_of_subnetworks
    K = deploy_param.n_subchannel
    Channel_gain = np.zeros([number_of_snapshots, K, N, N])
    PL = np.zeros([number_of_snapshots, N, N])
    dLoc = np.zeros([number_of_snapshots, N, 2])
    dLoc2 = np.zeros([number_of_snapshots, N, 2])

    dist, gwLoc, dvLoc = create_layout(deploy_param) # <- dung sensor static
    for i in tqdm(range(number_of_snapshots)):
        Channel_gain[i, :, :, :], PL[i, :, :] = compute_power(deploy_param, dist, gwLoc, K)
        dLoc[i, :, :] = gwLoc
        dLoc2[i, :, :] = dvLoc
    return Channel_gain, dLoc, dLoc2
