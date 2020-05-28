import os
import h5py
import numpy as np
from numpy.random import uniform, power, randint
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise.reproduceable import colored_noise
from pycbc.filter import resample_to_delta_t

def generate(file_path, duration, seed=0, signal_separation=200,
             signal_separation_interval=20, min_mass=1.2, max_mass=1.6,
             f_lower=20, srate=4096, padding=256, tstart=0):
    """Function that generates test data with injections.
    
    Arguments
    ---------
    file_path : str
        The path at which the data should be stored. 
    duration : int or float
        Duration of the output file in seconds.
    seed : {int, 0}, optional
        A seed to use for generating injection parameters and noise.
    signal_separation : {int or float, 200}, optional
        The average duration between two injections.
    signal_separation_interval : {int or float, 20}, optional
        The duration between two signals will be signal_separation + t,
        where t is drawn uniformly from the interval
        [-signal_separation_interval, signal_separation_interval].
    min_mass : {float, 1.2}, optional
        The minimal mass at which injections will be made (in solar
        masses).
    max_mass : {float, 1.6}, optional
        The maximum mass at which injections will be made (in solar
        masses).
    f_lower : {int or float, 20}, optional
        Noise will be generated down to the specified frequency.
        Below they will be set to zero. (The waveforms are generated
        with a lower frequency cutofff of 25 Hertz)
    srate : {int, 4096}, optional
        The sample rate at which the data is generated.
    padding : {int or float, 256}, optional
        Duration in the beginning and end of the data that does not
        contain any injections.
    tstart : {int or float, 0}, optional
        The inital time of the data.
    """    
    np.random.seed(seed)
    
    size = (duration // signal_separation)
    
    #Generate injection times
    random_time_samples = int(round(float(signal_separation_interval) * float(srate)))
    signal_separation_samples = int(round(float(signal_separation) * float(srate)))
    time_samples = randint(signal_separation_samples - random_time_samples, signal_separation_samples + random_time_samples, size=size)
    time_samples = time_samples.cumsum()
    times = time_samples / float(srate)
    
    times = times[np.where(np.logical_and(times > padding, times < duration - padding))[0]]
    size = len(times)
    
    #Generate parameters
    cphase = uniform(0, np.pi*2.0, size=size)
    
    ra = uniform(0, 2 * np.pi, size=size)
    dec = np.arccos(uniform(-1., 1., size=size)) - np.pi/2
    inc = np.arccos(uniform(-1., 1., size=size))
    pol = uniform(0, 2 * np.pi, size=size)
    dist = power(3, size) * 400

    m1 = uniform(min_mass, max_mass, size=size)
    m2 = uniform(min_mass, max_mass, size=size)
    
    #Save parameters to file.
    stat_file_path, ext = os.path.splitext(file_path)
    stat_file_path = stat_file_path + '_stats' + ext
    with h5py.File(stat_file_path, 'w') as f:
        f['times'] = times
        f['cphase'] = cphase
        f['ra'] = ra
        f['dec'] = dec
        f['inc'] = inc
        f['pol'] = pol
        f['dist'] = dist
        f['mass1'] = m1
        f['mass2'] = m2
        f['seed'] = seed
    
    p = aLIGOZeroDetHighPower(2 * int(duration * srate), 1.0/64, f_lower)
    
    #Generate noise
    data = {}
    for i, ifo in enumerate(['H1', 'L1']):
        data[ifo] = colored_noise(p, int(tstart),
                                    int(tstart + duration), 
                                    seed=seed + i,
                                    low_frequency_cutoff=f_lower)
        data[ifo] = resample_to_delta_t(data[ifo], 1.0/srate)
    
    # make waveforms and add them into the noise
    for i in range(len(times)):
        hp, hc = get_td_waveform(approximant="TaylorF2",
                                mass1=m1[i], 
                                mass2=m2[i],
                                f_lower=25,
                                delta_t=1.0/srate,
                                inclination=inc[i],
                                coa_phase=cphase[i],
                                distance=dist[i])
        hp.start_time += times[i] + int(tstart)
        hc.start_time += times[i] + int(tstart)
        
        for ifo in ['H1', 'L1']:
            ht = Detector(ifo).project_wave(hp, hc, ra[i], dec[i], pol[i])
            time_diff = float(ht.start_time - data[ifo].start_time)
            sample_diff = int(round(time_diff / data[ifo].delta_t))
            ht.prepend_zeros(sample_diff)
            ht.start_time = data[ifo].start_time
            data[ifo] = data[ifo].add_into(ht)
    
    #Save the data
    for ifo in ['H1', 'L1']:
        data[ifo].save(file_path, group='%s' % (ifo))
