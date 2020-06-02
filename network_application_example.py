from pycbc.types import TimeSeries, load_timeseries
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t
import numpy as np
import keras
import os
from test_data import generate
import matplotlib.pyplot as plt

def evaluate_ts_from_generator(network, generator, **kwargs):
    """Returns a SNR and a p-score TimeSeries, by applying the network
    to a provided generator.
    
    Arguments
    ---------
    network : keras.models.Model
        A Keras model with at least 2 outputs, where the first output is
        a scalar and the second output is a tuple of at least size 2.
    generator : keras.utils.Sequence
        A Keras generator that provides the input data for the network.
        The generator needs to have a attribute ts, which needs to be a
        list of pycbc.TimeSeries and an attribute time_step, which needs
        to be a float and is interpreted as second.
    verbose : {0 or 1 or 2, 1}
        Specifies how much output is generated during evaluation. See
        the documentation of keras.models.Model.predict_generator for
        details.
    workers : {int, 0}
        The number of parallel workers which should be used for
        evaluation. See the documentation of
        keras.models.Model.predict_generator for details.
    **kwargs:
        All other key-word arguments are passed to
        keras.models.Model.predict_generator. See the according
        documentation for details.
    
    Returns
    -------
    snr_ts : pycbc.TimeSeries
        A TimeSeries which is interpreted as the SNR estimate of the
        network. It has a delta_t equivalent to the step-size of the
        sliding window and the start time is set to the beginning of the
        input data.
    psc_ts : pycbc.TimeSeries
        A TimeSeries which is interpreted as the p-score estimate of the
        network. It has a delta_t equivalent to the step-size of the
        sliding window and the start time is set to the beginning of the
        input data.
    """
    if 'verbose' not in kwargs:
        kwargs['verbose'] = 1
    if 'workers' not in kwargs:
        kwargs['workers'] = 0
    res = network.predict_generator(generator, **kwargs)
    
    snr_ts = TimeSeries(res[0].flatten(), delta_t=generator.time_step, epoch=generator.ts[0]._epoch)
    bool_ts = TimeSeries([pt[0] for pt in res[1]], delta_t=generator.time_step)
    
    return snr_ts, bool_ts

def whiten_data(strain_list, low_freq_cutoff=20., max_filter_duration=4., psd=None):
    """Returns the data whitened by the PSD.
    
    Arguments
    ---------
    strain_list : list of pycbc.TimeSeries or pycbc.TimeSeries
        The data that should be whitened.
    low_freq_cutoff : {float, 20.}
        The lowest frequency that is considered during calculations. It
        must be >= than the lowest frequency where the PSD is not zero.
        Unit: hertz
    max_filter_duration : {float, 4.}
        The duration to which the PSD is truncated to in the
        time-domain. The amount of time is removed from both the
        beginning and end of the input data to avoid wrap-around errors.
        Unit: seconds
    psd : {None or pycbc.FrequencySeries, None}
        The PSD that should be used to whiten the data. If set to None
        the pycbc.psd.aLIGOZeroDetHighPower PSD will be used. If a PSD
        is provided which does not fit the delta_f of the data, it will
        be interpolated to fit.
    
    Returns
    -------
    list of pycbc.TimeSeries or TimeSeries
        Depending on the input type it will return a list of TimeSeries
        or a single TimeSeries. The data contained in this time series
        is the whitened input data, where the inital and final seconds
        as specified by max_filter_duration are removed.
    """
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
    """A Keras generator that takes a list of pycbc TimeSeries as input
    and generates the correctly whitened and formatted input to the
    network.
    
    Arguments
    ---------
    ts : list of pycbc.TimeSeries or pycbc.TimeSeries
        List of TimeSeries with the same duration and sample rate.
    time_step : {float, 0.25}
        The step-size with which the sliding window moves in seconds.
    batch_size : {int, 32}
        The batch-size to use, i.e. how many subsequent windows will be
        returned for each call of __getitem__.
    dt : {None or float, None}
        If ts is not a list of pycbc.TimeSeries but rather a list of
        arrays, the array will be cast to a TimeSeries with delta_t = dt.
    """
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
        
        #Number samples in each channel
        self.final_data_samples = 2048
        
        #delta_t of all TimeSeries
        self.dt = self.dt[0]
        
        #How big is the window that is shifted over the data
        #(64s + 8s for cropping when whitening)
        self.window_size_time = 72.0
        
        #Window size in samples
        self.window_size = int(self.window_size_time / self.dt)
        
        #How many points are shifted each step
        self.stride = int(self.time_step / self.dt)
        
        #total number of window shifts
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size + self.stride) / self.stride))
        
        #Different parts of the signal are re-sampled to different
        #delta_t. This lists the target delta_t.
        self.resample_dt = [1.0 / 4096, 1.0 / 2048, 1.0 / 1024,
                               1.0 / 512, 1.0 / 256, 1.0 / 128]
        
        #The inverse of the re-sample delta_t
        self.resample_rates = [4096, 4096, 2048, 1024, 512, 256, 128]
        
        #PSD used to whiten the data. Calculate once to save
        #computational resources.
        self.psd = aLIGOZeroDetHighPower(self.window_size//2+1, delta_f=1./self.window_size_time, low_freq_cutoff=18.)
    
    def __len__(self):
        """Returns the number of batches provided by this generator.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            Number of batches contained in this generator.
        """
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        """Return batch of index.
        
        Arguments
        ---------
        index : int
            The index of the batch to retrieve. It has to be smaller
            than len(self).
        
        Returns
        -------
        list of arrays
            A list containing the input for the network. Each array is
            of shape (batch_size, 2048, 2).
        """
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        
        #Check for last batch and adjust size if necessary
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        
        #Calculate the indices of the slices
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        #Generate correctly formatted input data
        X = self._gen_slices(index_range)
        
        return X
    
    def _gen_slices(self, index_range):
        """Slice, whiten and re-sample the input data.
        
        Arguments
        ---------
        index_range : numpy.array
            Array of shape (num_samples, 2), where each row contains the
            start- and stop-index of the slice.
        
        Returns
        -------
        list of arrays
            A list containing the input for the network. Each array is
            of shape (num_samples, 2048, 2).
        """
        
        #Number of different sample rates
        num_channels = len(self.resample_rates)
        
        #Number of detectors
        num_detectors = len(self.ts)
        
        #Setup return data
        X = [np.zeros((len(index_range), num_detectors * num_channels, self.final_data_samples)) for i in range(2)]
        X_ret = [np.zeros((num_detectors, self.final_data_samples, len(index_range))) for i in range(2*num_channels)]
        
        #Slice the time-series
        for in_batch, idx in enumerate(index_range):
            for detector in range(num_detectors):
                low, up = idx
                #Whiten each slice
                white_full_signal = whiten_data(self.ts[detector][low:up], psd=self.psd)
                
                #Re-sample each slice
                max_idx = len(white_full_signal)
                min_idx = max_idx - int(float(self.final_data_samples) / float(self.resample_rates[0]) / self.dt)
                for i, sr in enumerate(self.resample_rates):
                    X[0][in_batch][i * num_detectors + detector] = resample_to_delta_t(white_full_signal[min_idx:max_idx], 1.0 / sr)
                    if not i + 1 == num_channels:
                        t_dur = float(self.final_data_samples) / float(self.resample_rates[i+1])
                        sample_dur = int(t_dur / self.dt)
                        max_idx = min_idx
                        min_idx -= sample_dur
        
        #Change formatting
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
        print("Generating injections...")
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
    fig, axs = plt.subplots(2)
    axs[0].plot(snr_ts.sample_times, snr_ts)
    axs[0].grid()
    axs[0].set_xlabel('Time in s')
    axs[0].set_ylabel('predicted SNR')
    axs[1].plot(psc_ts.sample_times, psc_ts)
    axs[1].grid()
    axs[1].set_xlabel('Time in s')
    axs[1].set_ylabel('predicted p-score')
    fig.subplots_adjust(hspace=0.5)
    plt.savefig('example_results.png')
    plt.show()

if __name__ == "__main__":
    main()
