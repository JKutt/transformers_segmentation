#################################################
# Imports
import numpy as np
from scipy import fftpack
# from scipy import sparse
from scipy.special import factorial
# import properties
import scipy.signal as sig
from scipy.optimize import curve_fit

##################################################
# EMools Version 0.3  March 2022


class baseKernel(object):

    kernel = np.zeros(1)

    num_ensembles = 0

    reject_std = 1.0

    def __init__(self, filtershape=None, **kwargs):
        if filtershape is not None:
            if isinstance(filtershape, np.ndarray):
                self.filtershape = filtershape
                self.createFilterKernel()

    def createFilterKernel(self):
        """
        creates the filter kernel

        """
        if len(self.filtershape) > 1:
            # create the filter kernel
            tkernel = np.ones((3, 1))
            tkernel[1] = -2.0                    # 3 point kernel
            bsgn = np.ones((1, self.filtershape.size))
            bsgn[0, 1::2] = bsgn[0, 1::2] * -1   # alternate pol rem lin drift
            bwtd = np.matmul(tkernel,
                             bsgn * self.filtershape)  # filter weights
            tmp1 = np.arange(1, 4)
            tmp1 = np.reshape(tmp1, (3, 1))
            tmp2 = np.ones((1, self.filtershape.size))
            tmp3 = np.ones((3, 1))
            tmp4 = np.arange(self.filtershape.size)
            tmp4 = np.reshape(tmp4, (1, self.filtershape.size))
            knew = (np.matmul(tmp1, tmp2) +
                    np.matmul(tmp3, (tmp4 * (self.filtershape.size + 3))))
            btmp = np.zeros((self.filtershape.size + 2,
                             self.filtershape.size))  # create zero matrix
            shape_knew = knew.shape
            num_elements_kn = shape_knew[0] * shape_knew[1]
            knew = np.reshape(knew, num_elements_kn, order='F')
            shape = btmp.shape
            num_elements = shape[0] * shape[1]
            btmp = np.reshape(btmp, num_elements, order='F')
            shape_bwtd = bwtd.shape
            num_elements_b = shape_bwtd[0] * shape_bwtd[1]
            bwtd = np.reshape(bwtd, num_elements_b, order='F')
            for idx in range(knew.size):
                btmp[int(knew[idx]) - 1] = bwtd[idx]  # fill diag w/ weights
            btmp = np.reshape(btmp, shape, order='F')
            tHK = np.sum(btmp, 1)
            norm_tHK = np.sum(np.abs(tHK))
            tHK = tHK / norm_tHK
            tHK = np.reshape(tHK, (tHK.size, 1))
        else:
            raise Exception('filter size must be greater than 3!')
            tHK = np.zeros(1)
        self.kernel = tHK[:self.filtershape.size]         # assign weighted kernel

    def __mul__(self, val):
        """
        stack results

        """
        if isinstance(val, np.ndarray):
            return self._transform(val)
        else:
            raise Exception("Input must be an numpy array")

    @property
    def sizeOfFilter(self):
        """
            :rtypr int
            :return: number of points in filter kernel
        """
        return self.filtershape.size

    @property
    def getFilterKernel(self):
        """
           returns the filter kernel
        """
        return self.kernel


class statisticalRejectionKernel(baseKernel):
    """
       statistical rejection kernel for calculating
       a stack
    """

    def __init__(self,
                 sample_per_stack,
                 ip_start_time,
                 ip_end_time,
                 time,
                 **kwargs):
        baseKernel.__init__(self,
                            sample_per_stack,
                            **kwargs)

        self.sample_per_stack = sample_per_stack
        self.ip_start_time = ip_start_time
        self.ip_end_time = ip_end_time
        self.time = time
        self.tol = 1

    def _transform(self, dipole_time_series):
        """
        takes in stack data and returns decay
        Input:
        stack = Half period stacked Voltage data

        """
        # ========================================================================
        # level the time series
        #

        # trim to odd number of stacks
        num_stacks = np.floor(dipole_time_series.shape[0] / self.sample_per_stack)

        # determing number of samples for an odd amount of halh periods
        num_samples = int(num_stacks * self.sample_per_stack)

        time_s = self.time

        stacks = np.reshape(dipole_time_series, (int(self.sample_per_stack), int(num_stacks)), order='F')
        tstacks = np.reshape(time_s, (int(self.sample_per_stack), int(num_stacks)), order='F')

        pts = stacks[580, :]
        tpts = tstacks[580, :]

        time_ss = (tpts - time_s[0]) * 86400

        popt, _ = curve_fit(objective, time_ss, pts)

        # summarize the parameter values
        a, b, c, d, e, f = popt

        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(time_ss), max(time_ss), 1)
        # calculate the output for the range
        new_y = objective((time_s - time_s[0]) * 86400, a, b, c, d, e, f)

        # create the leveled time_series
        leveled_time_series = dipole_time_series - new_y

        # new leveled orginsed stack
        leveled_stacks = np.reshape(leveled_time_series, (int(self.sample_per_stack), int(num_stacks)), order='F')

        # correct polarity for the even -ve cycles
        leveled_stacks = leveled_stacks[:, 1::2] * -1

        # create the new time points to better work with fitting functions
        secondary_voltage = leveled_stacks[307:580, 0]

        time = np.arange(0, secondary_voltage.shape[0], 1)

        # initiate the standard deviation placeholder for each half cycle
        std = []

        # loop through and get the standard deviation
        for ii in range(leveled_stacks.shape[1]):

            secondary_voltage = leveled_stacks[self.ip_start_time:self.ip_end_time, ii]

            popt, _ = curve_fit(exponetial_decay_estimate, time, secondary_voltage)

            a, b, c = popt

            exp_fit = exponetial_decay_estimate(time, a, b, c)

            std += [np.std(secondary_voltage - exp_fit)]

        # now we find where standard deviation was higher than the mean
        mean_std = np.median(std) * self.tol
        indices_accepted = np.where(std < mean_std)

        # now put together the the accepted stacks
        accepted_stacks = leveled_stacks[:, indices_accepted[0]]

        resulting_stack = np.sum(accepted_stacks, axis=1) / accepted_stacks.shape[1]

        return resulting_stack, indices_accepted[0]


class decayKernel(baseKernel):
    """
       Decay kernel for calculating a decay
       from a stack
    """

    def __init__(self, num_windows=None,
                 window_starts=None,
                 window_ends=None,
                 window_weight=None,
                 window_overlap=None,
                 timebase=None,
                 output_type=None, **kwargs):
        baseKernel.__init__(self, None, **kwargs)

        if num_windows is not None:

            self.num_window = int(num_windows)
            self.window_starts = window_starts
            self.window_ends = window_ends
            self.window_weight = window_weight
            self.window_overlap = window_overlap
            self.output_type = output_type
            self.timebase = timebase

        else:

            raise Exception("need number of windows: num_windows")
        # self.window_overlap = self.window_overlap / self.divsor

    def _transform(self, stack):
        """
        takes in stack data and returns decay
        Input:
        stack = Half period stacked Voltage data

        """
        # calculate weighted window decay data =============
        if isinstance(stack, np.ndarray):
            # print(self.output_type)
            starts = self.getWindowStarts()

            if self.timebase is None:

                timebase = np.round(
                    starts[self.num_window - 1] / 1000.)
                timebase = timebase * 1000.

            else:
                timebase = self.timebase

            time = np.arange(0, timebase, (timebase / stack.size))
            vsDecay = np.zeros((self.num_window))
            vs_std = np.zeros((self.num_window))

            # loop through and do every window
            for win in range(self.num_window):

                self.window_starts[win]
                # find how many samples in first window
                cntr = 0

                # get time span for tail of windows
                start_tmp = self.window_starts[win] - (
                    self.window_overlap *
                    (self.window_ends[win] - self.window_starts[win]))

                if win == (vsDecay.size - 1):

                    end_tmp = self.window_ends[win]

                else:

                    end_tmp = self.window_ends[win] + (
                        self.window_overlap *
                        (self.window_ends[win] - self.window_starts[win]))
                # print start_tmp, end_tmp

                for i in range(stack.size):
                    if time[i] >= start_tmp and time[i] <= end_tmp:
                        cntr += 1

                # create window wieghts
                indx1 = np.arange(0, self.window_weight)
                weights = 0.5 - (0.5 *
                                 np.cos((2 * np.pi * indx1) /
                                        (indx1.size - 1)))
                # create new weights
                Wg = np.zeros(cntr)
                start_Wg = (indx1.size / 2.0 - 1.0) - (cntr / 2.0) + 1

                for r in range(Wg.size):
                    Wg[r] = weights[int(start_Wg) + r]

                # create vector storing weighted values
                Vs_window = np.zeros(cntr)
                Vs_window_ave = np.zeros(cntr)
                # get window times
                w_idx = np.zeros(cntr)

                # assign total time and time step
                count = 0
                for i in range(time.size):

                    if time[i] >= start_tmp and time[i] <= end_tmp:

                        w_idx[count] = time[i]
                        Vs_window[count] = stack[i] * Wg[count]
                        Vs_window_ave[count] = stack[i]
                        count += 1

                sumWin = np.sum(Vs_window)      # sum the values of the window
                vs_std[win] = np.std(Vs_window_ave)  # standard deviation of window
                # print Vs_window
                vsDecay[win] = sumWin / cntr
        else:

            raise Exception("input must be a stack numpy array!")
            vsDecay = np.zeros(1)
        # end decay =======================================

        output = vsDecay
        # output = vs_std
        if self.output_type == 'std':
            output = vs_std

        return output

    def getWindowStarts(self):
        return self.window_starts

    def getWindowCenters(self):
        """
           returns window centers
        """
        window_centers = (np.asarray(self.window_ends) +
                          np.asarray(self.window_starts)) / 2.
        return window_centers

    def getWindowWidths(self):
        """
           returns window widths
        """
        window_widths = (np.asarray(self.window_ends) -
                         np.asarray(self.window_starts))
        return window_widths


class frequencyWindowingKernel(baseKernel):
    """
       Frequency windowing kernel for evaluating frequencies
       from a time-series recording
    """

    def __init__(self,
                 eval_frequencies=None,
                 window_overlap=None,
                 output_type=None, **kwargs):
        baseKernel.__init__(self, None, **kwargs)

        if eval_frequencies is not None:
            self.eval_frequencies = eval_frequencies
            self.num_window = self.eval_frequencies.size
            self.window_overlap = window_overlap
            self.output_type = output_type
        else:
            raise Exception("need eval. frequencies: eval_frequencies")
        # self.window_overlap = self.window_overlap / self.divsor

    def _transform(self, time_series):
        """
        takes in stack data and returns decay
        Input:
        stack = Half period stacked Voltage data

        """
        # calculate weighted window decay data =============
        if isinstance(time_series, np.ndarray):
            # print(self.output_type)
            if time_series.shape.size > 1:         # check mxn - stack results
                print("stacking similar frequency windows")
            else:                                  # continue with windowing
                print("windowing a single time-series")
            # starts = self.getWindowStarts()
            # # create window wieghts
            # indx1 = np.arange(0, self.window_weight)
            # # slepian goes here
            # window = signal.slepian(51, width=0.3)
            # weights = 0.5 - (0.5 *
            #                  np.cos((2 * np.pi * indx1) /
            #                         (indx1.size - 1)))
            # # create new weights
            # Wg = np.zeros(cntr)
            # start_Wg = (indx1.size / 2.0 - 1.0) - (cntr / 2.0) + 1
            # for r in range(Wg.size):
            #     Wg[r] = weights[int(start_Wg) + r]
            # # print Wg
            # # create vector storing weighted values
            # Vs_window = np.zeros(cntr)
            # Vs_window_ave = np.zeros(cntr)
            # # get window times
            # w_idx = np.zeros(cntr)
            # # assign total time and time step
            # count = 0
            # for i in range(time.size):
            #     if time[i] >= start_tmp and time[i] <= end_tmp:
            #         w_idx[count] = time[i]
            #         Vs_window[count] = stack[i] * -1 * Wg[count]
            #         Vs_window_ave[count] = stack[i] * -1
            #         count += 1
            # sumWin = np.sum(Vs_window)      # sum the values of the window
            # vs_std[win] = np.std(Vs_window_ave)  # standard deviation of window
            # # print Vs_window
            # vsDecay[win] = sumWin / cntr
        else:
            raise Exception("input must be a stack numpy array!")
            vsDecay = np.zeros(1)
            # end decay =======================================
        output = vsDecay
        # output = vs_std
        if self.output_type == 'std':
            output = 0 #vs_std

        return output

    def getWindowStarts(self):
        return self.window_starts

    def getWindowWidths(self):
        """
           returns window widths
        """
        window_widths = (np.asarray(self.window_ends) -
                         np.asarray(self.window_starts))
        return window_widths


class filterKernel(baseKernel):
    """
        Filter Kernel for stacking a time-series
        raw signal
    """

    def __init__(self, filtershape, **kwargs):
        baseKernel.__init__(self, filtershape, **kwargs)
        # self.createFilterKernel()            # create filter kernel

    def _transform(self, signal):
        """
           performs the stacking calculation
        """
        size_of_stack = int(signal.size / (self.kernel.size))

        Ax = np.reshape(signal, (int(size_of_stack),
                        int(self.kernel.size)), order='F')
        shape_Ax = Ax.shape
        shape_tHK = self.kernel.shape
        if shape_Ax[1] == shape_tHK[0]:
            stack = np.matmul(Ax, self.kernel)  # create stack data
            return stack
        else:
            return 0


class ensembleKernel(baseKernel):
    """
    ensemble stacking kernel. Each ensemble overlaps by defined amount

    TODO: Add some statistical rejection between the ensembles

    """

    def __init__(self, filtershape,
                 number_half_periods, **kwargs):

        filter_kernel = createHanningWindow(filtershape.size)
        baseKernel.__init__(self, filter_kernel, **kwargs)
        self.number_half_periods = number_half_periods
        self.kernel_ends = self.createFilterKernelEnds(filtershape.size)

    def createFilterKernelEnds(self, ensemble_size):
        """
        creates the filter kernel

        """
        ends_window = createHanningWindow(ensemble_size - 2)
        # print(ends_window.size)
        if ends_window.size > 1:
            # create the filter kernel
            tkernel = np.ones((3, 1))
            tkernel[1] = -2.0                    # 3 point kernel
            bsgn = np.ones((1, ends_window.size))
            bsgn[0, 1::2] = bsgn[0, 1::2] * -1   # alternate pol rem lin drift
            bwtd = np.matmul(tkernel,
                             bsgn * ends_window)  # filter weights
            tmp1 = np.arange(1, 4)
            tmp1 = np.reshape(tmp1, (3, 1))
            tmp2 = np.ones((1, ends_window.size))
            tmp3 = np.ones((3, 1))
            tmp4 = np.arange(ends_window.size)
            tmp4 = np.reshape(tmp4, (1, ends_window.size))
            knew = (np.matmul(tmp1, tmp2) +
                    np.matmul(tmp3, (tmp4 * (ends_window.size + 3))))
            btmp = np.zeros((ends_window.size + 2,
                             ends_window.size))  # create zero matrix
            shape_knew = knew.shape
            num_elements_kn = shape_knew[0] * shape_knew[1]
            knew = np.reshape(knew, num_elements_kn, order='F')
            shape = btmp.shape
            num_elements = shape[0] * shape[1]
            btmp = np.reshape(btmp, num_elements, order='F')
            shape_bwtd = bwtd.shape
            num_elements_b = shape_bwtd[0] * shape_bwtd[1]
            bwtd = np.reshape(bwtd, num_elements_b, order='F')
            for idx in range(knew.size):
                btmp[int(knew[idx]) - 1] = bwtd[idx]  # fill diag w/ weights
            btmp = np.reshape(btmp, shape, order='F')
            tHK = np.sum(btmp, 1)
            norm_tHK = np.sum(np.abs(tHK))
            tHK = tHK / norm_tHK
            tHK = np.reshape(tHK, (tHK.size, 1))
        else:
            raise Exception('filter size must be greater than 3!')
            tHK = np.zeros(1)
        return tHK

    def _transform(self, signal):
        """
           performs the stacking calculation using Ensembles
        """
        sub_signals = []
        sub_samples = []
        size_of_stack = int(signal.size / (self.number_half_periods))
        overlap = 2                                                        # hard code standard
        T_per_ensemble = (self.kernel.size - overlap)                 # desired half T

        number_of_ensembles = int(np.floor(signal.size / (T_per_ensemble * size_of_stack)))
        opt_number_of_samples = int(number_of_ensembles * T_per_ensemble * size_of_stack)
        signal = signal[:opt_number_of_samples]
        ensembles = np.zeros((size_of_stack, number_of_ensembles))         # matrix holding all the stacks

        for index in range(number_of_ensembles):

            if index == 0:

                T_overlap = T_per_ensemble + overlap
                end_index = T_overlap * size_of_stack
                trim_signal = signal[:end_index]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(0, end_index))
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(T_overlap)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel_ends.shape

                if shape_Ax[1] == shape_tHK[0]:
                    stack = np.matmul(Ax, self.kernel_ends)  # create stack data
                    ensembles[:, index] = stack.T
                else:
                    print("fail stack, wrong size")

            elif index == 1:

                T_overlap = T_per_ensemble + overlap
                start_index = size_of_stack * (T_per_ensemble - 2)
                end_index = start_index + (T_overlap * size_of_stack)
                trim_signal = signal[start_index:end_index]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(start_index, end_index))
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernel.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel.shape

                if shape_Ax[1] == shape_tHK[0]:

                    stack = np.matmul(Ax, self.kernel)  # create stack data
                    ensembles[:, index] = stack.T

                else:

                    print("fail stack, wrong size")

            elif index == (number_of_ensembles - 1):

                T_overlap = T_per_ensemble + overlap                   # get end overlap
                start_index = (index * (T_per_ensemble) - 2) * size_of_stack
                trim_signal = signal[start_index:]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(start_index, signal.size))
                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernel.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel.shape

                if shape_Ax[1] == shape_tHK[0]:

                    stack = np.matmul(Ax, self.kernel)  # create stack data
                    ensembles[:, index] = stack.T

                else:

                    print("fail stack, wrong size in last")

            else:

                T_overlap = T_per_ensemble + overlap
                start_index = (((index) * T_per_ensemble) - 2) * size_of_stack
                end_index = start_index + (T_overlap * size_of_stack)
                trim_signal = signal[start_index:end_index]
                sub_signals.append(trim_signal)
                sub_samples.append(np.arange(start_index, end_index))

                Ax = np.reshape(trim_signal, (int(size_of_stack),
                                int(self.kernel.size)), order='F')
                shape_Ax = Ax.shape
                shape_tHK = self.kernel.shape

                if shape_Ax[1] == shape_tHK[0]:

                    stack = np.matmul(Ax, self.kernel)  # create stack data
                    ensembles[:, index] = stack.T

                else:

                    print("fail stack, wrong size")

        ensembles[:, 1::2] = ensembles[:, 1::2] * -1   # make all stacks same polarity

        return ensembles, sub_signals, sub_samples


##################################################
# define methods
##################################################

def exponetial_decay_estimate(x, a, b, c):

    return a * np.exp(-x ** b) + c

def objective(x, a, b, c, d, e, f):

    return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + f


def getFFT(signal):
    """
       :rtype numpy array
       :return: fft of the filter kernel
    """
    v_fft = fftpack.fft(signal)
    return v_fft[0:int(v_fft.size / 2 - 1)] / np.max(v_fft)


def getPSD(signal, sample_rate=150, samples_per_segment=512):
    """
       :rtype numpy array
       :return: Power Spectral Density and sampled frequencies
    """
    f, psd = sig.welch(signal,
                       fs=sample_rate,
                       window='blackman',
                       nperseg=samples_per_segment,
                       scaling='spectrum')

    return f, psd


def getFrequnceyResponse(signal):
    """
       :rtype numpy array
       :return: frequeny response of the filter kernel
    """
    v_fft = fftpack.fft(signal)
    amplitude = np.sqrt(v_fft.real**2 + v_fft.imag**2)
    return amplitude[0:(int(amplitude.size / 2 - 1))] / np.max(amplitude)


def getCrossCorrelation(signal1, signal2):
    """
       :rtype numpy array
       :return: frequeny response of the filter kernel
    """
    x_corr = np.correlate(signal1, signal2, mode='full')
    return x_corr / np.max(np.abs(x_corr))


def getCoherence(signal1, signal2, sample_rate):
    f, coherence = sig.coherence(signal1, signal2, sample_rate, nperseg=1024)
    return f, coherence


def getSpectralDensity(signal):
    """
       X_d(f)X_d^*(f)
    """
    signal_f = fftpack.fft(signal)
    signal_f_conj = np.conjugate(signal_f)
    S_xx = signal_f * signal_f_conj
    # return S_xx[0:int(S_xx.size / 2 - 1)] / np.max(np.abs(S_xx))
    return S_xx


def getPhaseResponse(signal):
    """
       :rtype numpy array
       :return: Rhase response of the filter kernel
    """
    v_fft = fftpack.fft(signal)
    phase = np.arctan2(v_fft.imag, v_fft.real)
    return phase[0:(phase.size / 2 - 1)]


def getCWT(sig, wavelet=None, widths=None):
    if wavelet is not None:
        cwtmatr = sig.cwt(sig, sig.ricker, widths)
    return cwtmatr


def padNextPower2(ts, num_ensembles=1):
    """
    Input: numpy time-series
    returns: a zeros padded time-series
    rtype: numpy array
    - if ensemble option is chosen, the time-series is arranged
      in a nxm numpy array

    """
    if num_ensembles == 1:
        next_2 = 1 << (ts.size - 1).bit_length()
        diff_in_size = next_2 - ts.size
        pad = np.zeros(diff_in_size)
        return np.concatenate((ts, pad), axis=0)
    elif num_ensembles > 1:
        return None
    else:
        return None


def getWeightedVs(stack, window_start, window_end, attenuation):
    """
    takes in data and returns decay
    Input:
    stack = Half period stacked Voltage data
    window_start = an array of window start times
    window_end = an array of window end times

    """
    timebase = np.round(window_end[len(window_end) - 1] / 1000.0)
    timebase = timebase * 1000
    time = np.arange(0, timebase, (timebase / stack.size))
    vsDecay = np.zeros((len(window_end)))
    # loop through and do every window
    for win in range(len(window_start)):
        # find how many samples in first window
        cntr = 0
        # get time span for tail of windows
        start_tmp = window_start[win] - (
            0.45 * (window_end[win] - window_start[win]))
        if win == (vsDecay.size - 1):
            end_tmp = window_end[win]
        else:
            end_tmp = window_end[win] + (
                0.45 * (window_end[win] - window_start[win]))
        # print start_tmp, end_tmp
        for i in range(stack.size):
            if time[i] >= start_tmp and time[i] <= end_tmp:
                cntr += 1
        # create window wieghts
        indx1 = np.arange(0, attenuation)
        weights = 0.5 - (0.5 * np.cos((2 * np.pi * indx1) / (indx1.size - 1)))
        # create new weights
        Wg = np.zeros(cntr)
        start_Wg = (indx1.size / 2.0 - 1.0) - (cntr / 2.0) + 1
        for r in range(Wg.size):
            Wg[r] = weights[int(start_Wg) + r]
        # print Wg
        # create vector storing weighted values
        Vs_window = np.zeros(cntr)
        Vs_window_ave = np.zeros(cntr)
        # get window times
        w_idx = np.zeros(cntr)
        # assign total time and time step
        count = 0
        for i in range(time.size):
            if time[i] >= start_tmp and time[i] <= end_tmp:
                w_idx[count] = time[i]
                Vs_window[count] = stack[i] * Wg[count]
                Vs_window_ave[count] = stack[i]
                count += 1
        sumWin = np.sum(Vs_window)
        # print Vs_window
        vsDecay[win] = sumWin / cntr

    return vsDecay


def createBruteStackWindow(num_points):
    num_points = num_points - 6
    tmp = np.ones(num_points)
    tmp = tmp * 4

    # create full filter kernel
    f1 = np.zeros(tmp.size + 4)
    f1[0] = 1
    f1[1] = 3
    f1[f1.size - 2] = 3
    f1[f1.size - 1] = 1

    for j in range(tmp.size):
        f1[j + 2] = tmp[j]

    f1 = f1 / (4.0 * (num_points - 2))

    return f1


def createMexHatWavelet(num_points, a):
    return sig.ricker(num_points, a)


def createSlepianWindow(num_size, attenuation):
    return sig.slepian(num_size, width=0.3)


def createKaiserWindow(num_taps, attenuation):
    """
    creates a Kaiser window
    Input:
    num_taps = number of taps for the requested window

    """
    return sig.kaiser(num_taps, beta=attenuation)


def mbessel(position, max_iter):
    seq_m = np.arange(0, max_iter)
    fact_m = factorial(seq_m, exact=False)
    summation = 0.0
    for i in range(max_iter):
        inc = np.power(1.0 / fact_m * np.power(position * 0.5, i), 2)
        frac = inc / summation
        summation += inc
        if frac < 0.001:
            break
    return summation


def cheby_poly(n, x):
    pos = 0.0
    if np.abs(x) <= 1.0:
        pos = np.cos(n * np.arccos(x))
    else:
        pos = np.cosh(n * np.arccosh(x))
    return pos


def createHanningWindow(num_points):
    """
    creates a Hanning window filter kernel
    Input:
    num_points = number of taps for the requested window

    """
    # create  sequence array
    indx1 = np.arange(0, num_points).T
    # creates window
    filterWindow = 0.5 * (1 - np.cos((2 * np.pi /
                                      (indx1.size - 1)) *
                                     indx1))

    return filterWindow


def createChebyshevWindow(num_taps, attenuation):
    """
    creates a Chebyshev window
    Input:
    num_taps = number of taps for the requested window

    """

    return sig.chebwin(num_taps, at=attenuation)


def getPrimaryVoltageError(signal, num_stack=None, start=None, end=None):
    """
        Extracts the Voltage standard deviation of the signal.
        Input:
        start = percent of the on time to start calculation
        end = percent of on time to end calculation
        e.g 50% to 90%
        """

    if start and end and num_stack is not None:
        size_of_stack = int(signal.size / (num_stack))
        Ax = np.reshape(signal, (int(size_of_stack),
                                 int(num_stack)), order='F')
        sumStart = int((start / 100.0) * (num_stack / 2))  # start Vp calc
        sumEnd = int((end / 100.0) * (num_stack / 2))     # end of Vp calc
        std_of_each_ontime = np.std(Ax[sumStart:sumEnd, :], axis=0)
        return np.mean(std_of_each_ontime)
    else:
        print('[ERROR] Please supply a start and end time to calcualte stability')


def getPrimaryVoltage(start, end, stack):
        """
        Extracts the Vp of the signal.
        Input:
        start = percent of the on time to start calculation
        end = percent of on time to end calculation
        e.g 50% to 90%
        """

        sumStart = int((start / 100.0) * (stack.size / 2))  # start Vp calc
        sumEnd = int((end / 100.0) * (stack.size / 2))     # end of Vp calc
        Vp = np.sum(stack[sumStart:sumEnd]) / (sumEnd - sumStart)

        return Vp


def waveformGenerator(sample_rate=150, duty=50, frequency=None, num_periods=2):
    if frequency is None:
        print('Please supply a frequency!')
    else:
        eps = 1e-4
        # end_time = (1 / frequency) * num_periods
        T = 1 / frequency
        half_duty = int((sample_rate * T) / 4)
        t = np.arange(0, half_duty)
        xt = np.ones(half_duty * 4)
        vs = np.exp(-2 * np.pi * t / 2)
        xt[half_duty:2 * half_duty] = vs + eps
        xt[2 * half_duty:3 * half_duty] = -1
        xt[3 * half_duty:] = -1 * (vs + eps)

        xt_out = []
        for idx in range(num_periods):
            xt_out.append(xt)
        return np.hstack(xt_out)


def adjustTsToStartOfPeriod(xt, time, sample_rate=150, timebase=2000):
    xt = np.asarray(xt)
    time = np.asarray(time)
    samples_to_plot = int((timebase / 1000.) * (sample_rate * 2) * 4)
    limits_pos = np.max(xt[:samples_to_plot])
    peaks, _ = sig.find_peaks(xt[:samples_to_plot], height=limits_pos)
    peaks = int(peaks - 2)
    tmp_sig = xt[peaks:]
    num_periods = np.floor(tmp_sig.size /
                           (sample_rate * (timebase / 1e3) * 4))
    end_shift = int(peaks + num_periods * (sample_rate * (timebase / 1e3) * 4))
    new_signal = xt[peaks:end_shift]
    new_time = time[peaks:end_shift]
    return new_signal, new_time
