from pycbc.types import TimeSeries, load_timeseries
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t
import numpy as np
import keras
import os
from test_data import generate
import matplotlib.pyplot as plt
import h5py

def evaluate_ts_from_generator(network, generator, batch_size=32):
    res = network.predict_generator(generator, verbose=1, workers=0)
    
    snr_ts = TimeSeries(res[0].flatten(), delta_t=generator.time_step, epoch=generator.ts[0]._epoch)
    bool_ts = TimeSeries([pt[0] for pt in res[1]], delta_t=generator.time_step)
    
    return snr_ts, bool_ts

def whiten_data(strain_list, low_freq_cutoff=20., max_filter_duration=4., psd=None):
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    ret = []
    for strain in strain_list:
        df = strain.delta_f
        f_len = int(len(strain) / 2) + 1
        if psd is None:
            psd = aLIGOZeroDetHighPower(length=f_len, delta_f=df, low_freq_cutoff=low_freq_cutoff-2.)
        else:
            if not len(psd) == f_len:
                msg = 'Length of PSD does not match data.'
                raise ValueError(msg)
            elif not psd.delta_f == df:
                psd = interpolate(psd, df)
        max_filter_len = int(max_filter_duration * strain.sample_rate) #Cut out the beginning and end
        psd = inverse_spectrum_truncation(psd, max_filter_len=max_filter_len, low_frequency_cutoff=low_freq_cutoff, trunc_method='hann')
        f_strain = strain.to_frequencyseries()
        kmin = int(low_freq_cutoff / df)
        f_strain.data[:kmin] = 0
        f_strain.data[-1] = 0
        f_strain.data[kmin:] /= psd[kmin:] ** 0.5
        strain = f_strain.to_timeseries()
        ret.append(strain[max_filter_len:len(strain)-max_filter_len])
    
    if not org_type == list:
        return(ret[0])
    else:
        return(ret)

class time_series_generator(keras.utils.Sequence):
    def __init__(self, ts, time_step=0.25, batch_size=32, dt=None):
        self.batch_size = batch_size
        self.time_step = time_step
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        self.final_data_samples = 2048
        #The delta_t for all data
        self.dt = self.dt[0]
        #How big is the window that is shifted over the data
        #(64s + 8s for cropping when whitening)
        self.window_size_time = 72.0
        self.window_size = int(self.window_size_time / self.dt)
        #How many points are shifted each step
        self.stride = int(self.time_step / self.dt)
        #How many window shifts happen
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size + self.stride) / self.stride))
        
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128]
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128]
        self.num_samples = 2048
        
        self.psd = aLIGOZeroDetHighPower(self.window_size//2+1, delta_f=1./self.window_size_time, low_freq_cutoff=18.)
    
    def __len__(self):
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        X = self._gen_slices(index_range)
        
        return X
    
    def _gen_slices(self, index_range):
        num_channels = len(self.resample_rates)
        num_detectors = len(self.ts)
        X = [np.zeros((len(index_range), num_detectors * num_channels, self.final_data_samples)) for i in range(2)]
        X_ret = [np.zeros((num_detectors, self.final_data_samples, len(index_range))) for i in range(2*num_channels)]
        
        for in_batch, idx in enumerate(index_range):
            for detector in range(num_detectors):
                low, up = idx
                white_full_signal = whiten_data(self.ts[detector][low:up], psd=self.psd)
                max_idx = len(white_full_signal)
                min_idx = max_idx - int(float(self.num_samples) / float(self.resample_rates[0]) / self.dt)
                for i, sr in enumerate(self.resample_rates):
                    X[0][in_batch][i * num_detectors + detector] = resample_to_delta_t(white_full_signal[min_idx:max_idx], 1.0 / sr)
                    if not i + 1 == num_channels:
                        t_dur = float(self.num_samples) / float(self.resample_rates[i+1])
                        sample_dur = int(t_dur / self.dt)
                        max_idx = min_idx
                        min_idx -= sample_dur
        
        for i in range(2):
            X[i] = X[i].transpose(1, 2, 0)
        
        for i in range(num_channels):
            X_ret[2*i][0] = X[0][2*i]
            X_ret[2*i][1] = X[0][2*i+1]
            X_ret[2*i+1][0] = X[1][2*i]
            X_ret[2*i+1][1] = X[1][2*i+1]
        
        return([x.transpose(2, 1, 0) for x in X_ret])
    
    def on_epoch_end(self):
        return

def main():
    #Load network
    network = keras.models.load_model('network.hdf')
    
    #Generate and load data
    file_path = 'example_injections.hdf'
    if not os.path.isfile(file_path):
        generate(file_path, 1024)
    L1 = load_timeseries(file_path, group='L1')
    H1 = load_timeseries(file_path, group='H1')
    print("Loading complete.")
    
    #Apply the network
    generator = time_series_generator([L1, H1], time_step=0.25, batch_size=128)
    snr_ts, psc_ts = evaluate_ts_from_generator(network, generator)
    
    snr_ts.save('snr_time_series.hdf')
    psc_ts.save('p-score_time_series.hdf')
    
    #Plot the two resulting time series
    plt.plot(snr_ts.sample_times,  snr_ts)
    plt.show()

if __name__ == "__main__":
    main()
