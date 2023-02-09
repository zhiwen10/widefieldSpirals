import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert

def loadUVt(expRoot):
    movieSuffix = "blue"
    U = np.load(os.path.join(expRoot, movieSuffix, "svdSpatialComponents.npy"))
    mimg = np.load(os.path.join(expRoot, movieSuffix, "meanImage.npy"))
    corrPath = os.path.join(expRoot, 'corr', 'svdTemporalComponents_corr.npy')
    V = np.load(corrPath)
    t = np.load(os.path.join(expRoot, 'corr', 'svdTemporalComponents_corr.timestamps.npy'))
    return U, V, t, mimg


def loadEphys(ksDir):
    sample_rate = 30000
    ss = np.load(os.path.join(ksDir, 'spike_times.npy'))
    st = ss.astype('float64')/sample_rate
    spikeTemplates = np.load(os.path.join(ksDir, 'spike_templates.npy'))

    if os.path.exists(os.path.join(ksDir, 'spike_clusters.npy')):
        clu = np.load(os.path.join(ksDir, 'spike_clusters.npy'))
    else:
        clu = spikeTemplates

    tempScalingAmps = np.load(os.path.join(ksDir, 'amplitudes.npy'))

    if os.path.exists(os.path.join(ksDir, 'cluster_groups.csv')) :
        cgsFile = os.path.join(ksDir, 'cluster_groups.csv')

    if os.path.exists(os.path.join(ksDir, 'cluster_group.tsv')) :
        cgsFile = os.path.join(ksDir, 'cluster_group.tsv')

    cluster_table = pd.read_csv(cgsFile,sep="\t")

    cluster_table.group = pd.Categorical(cluster_table.group)
    cluster_table['code'] = cluster_table.group.cat.codes
    noiseClusters = cluster_table[cluster_table["code"]==2]
    noiseClusters["cluster_id"].to_numpy()
    cids = cluster_table["cluster_id"].to_numpy()
    cgs = cluster_table["code"].to_numpy()
    noise_indx = np.isin(clu,noiseClusters)
    st = st[~noise_indx]
    spikeTemplates = spikeTemplates[~noise_indx]
    tempScalingAmps = tempScalingAmps[~noise_indx]
    clu = clu[~noise_indx]
    cgs = cgs[~np.isin(cids,noiseClusters)]
    cids = cids[~np.isin(cids,noiseClusters)]
    coords = np.load(os.path.join(ksDir, 'channel_positions.npy'))
    ycoords = coords[:,1]
    xcoords = coords[:,0]
    temps = np.load(os.path.join(ksDir, 'templates.npy'))
    spikeStruct1 = spikeStruct(st, spikeTemplates, clu, tempScalingAmps, cgs, cids, xcoords, ycoords, temps)

    return spikeStruct1

class spikeStruct:
    def __init__(self, st, spikeTemplates, clu, tempScalingAmps, cgs, cids, xcoords, ycoords, temps):
        spikeStruct.st = st
        spikeStruct.spikeTemplates = spikeTemplates
        spikeStruct.clu = clu
        spikeStruct.tempScalingAmps = tempScalingAmps
        spikeStruct.cgs = cgs
        spikeStruct.cids = cids
        spikeStruct.xcoords = xcoords
        spikeStruct.ycoords = ycoords
        spikeStruct.temps = temps    

def filter_and_hilbert(U, dV):
    """
    U1 shape: x * y * 50 components
    dV1 shape: 50 components * time_samples
    """
    Ur = U.reshape(-1,50)
    trace = Ur@dV
    trace = trace.reshape(U.shape[0],U.shape[1],-1)
    sos = signal.butter(2, [2,8], 'bandpass', fs=35, output='sos')
    filtered = signal.sosfiltfilt(sos, trace,axis = 2)
    filtered_mean = np.mean(filtered,2)
    filtered_mean = filtered_mean[:,:, np.newaxis]
    filtered = filtered-filtered_mean
    analytic_signal = hilbert(filtered,axis=2)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal,deg=False)    
    return filtered, amplitude_envelope, instantaneous_phase    

def pad_zeros(trace_phase1,halfpadding):
    """
    tracePhase size: xsize * ysize * nframes
    """
    padding = 2* halfpadding
    xsize = trace_phase1.shape[0]
    ysize = trace_phase1.shape[1]
    nframe = trace_phase1.shape[2]
    trace_phase = np.zeros([xsize+padding ,ysize+padding,nframe])
    trace_phase[halfpadding:halfpadding+xsize,halfpadding:halfpadding+ysize,:] = trace_phase1
    return trace_phase

def check_spiral(trace_phase,px,py,r,th,spiral_range):
#     th = np.arange(0,360,36)
#     r = 20
#     px = 510
#     py = 400
#     spiral_range = np.linspace(-np.pi,np.pi,5)
    cx = np.round(r*np.cos(np.radians(th))+px).astype('int64')
    cy = np.round(r*np.sin(np.radians(th))+py).astype('int64')
    ph = trace_phase[cy,cx]
    ph2 = np.diff(ph)
    ph3 = wrapToPi(ph2)
    ph4 = np.zeros((ph3.shape[0],1))
    ph4[0] = ph3[0]
    for i in np.arange(1,ph3.shape[0]):   
        ph4[i] = ph4[i-1]+ph3[i]    
    angle_range = abs(ph4[-1]-ph4[0])               
    hist_n, bin_edges = np.histogram(ph, bins=spiral_range)
    spiral_temp = [px,py,r,0]
    if np.all(angle_range>5) & np.all(angle_range<7) & all(hist_n):
        spiral_temp[-1] = 1
    return spiral_temp

def wrapToPi(lambda1):
    lambda1 = (lambda1 + np.pi) % (2 * np.pi) - np.pi
    return lambda1

