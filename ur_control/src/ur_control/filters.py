# ROS utilities used by the CRI group
#! /usr/bin/env python
import rospy, math
import numpy as np
import scipy.signal


def best_fit_foaw(y, fs, m, d):
  """
  First-Order Adaptive Windowing (FOAW)

  Yet another differentiation filter.

  @type  y: list
  @param y: the values of the time history of the signal.
  @type  fs: float
  @param fs: The sampling frequency (Hz) of the signal to be filtered
  """
  T = 1.0/fs
  result = np.zeros(len(y))
  for k in range(len(y)):
    slope = 0
    for n in range(1, min(m,k)):
      # Calculate slope over interval (best-fit-FOAW)
      b = ( ( n*sum([y[k-i]     for i in range(n+1)])
            - 2*sum([y[k-i]*i   for i in range(n+1)]) )
          / (T*n*(n+1)*(n+2)/6) )
      # Check the linear estimate of each middle point
      outside = False
      for j in range(1,n):
        ykj = y[k]-(b*j*T)
        # Compare to the measured value within the noise margin
        # If it's outside noise margin, return last estimate
        #~ if abs(y[k-j] - ykj) > 2*d:
        if ykj < (y[k-j]-d) or ykj > (y[k-j]+d):
          outside = True
          break
      if outside: break
      slope = b
    result[k] = slope
  return result

def butter_lowpass(cutoff, fs, order=5):
  """
  Butterworth lowpass digital filter design.

  Check C{scipy.signal.butter} for further details.

  @type  cutoff: float
  @param cutoff: Cut-off frequency in Hz
  @type  fs: float
  @param fs: The sampling frequency (Hz) of the signal to be filtered
  @type  order: int
  @param order: The order of the filter.
  @rtype: b, a
  @return: Numerator (b) and denominator (a) polynomials of the IIR
  filter.
  """
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def lowpass_fo(cutoff, fs):
  """
  First-order Lowpass digital filter design.

  @type  cutoff: float
  @param cutoff: Cut-off frequency in Hz
  @type  fs: float
  @param fs: The sampling frequency (Hz) of the signal to be filtered
  @rtype: b, a
  @return: Numerator (b) and denominator (a) polynomials of the IIR
  filter.
  """
  import control.matlab
  F = control.matlab.tf(1,np.array([1/(2*np.pi*cutoff),1]))
  Fz = control.matlab.c2d(F, 1/fs, 'zoh')
  b = Fz.num[0][0][-1]
  a = Fz.den[0][0]
  return b, a

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
  """
  Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

  The Savitzky-Golay filter removes high frequency noise from data.
  It has the advantage of preserving the original shape and
  features of the signal better than other types of filtering
  approaches, such as moving averages techniques.

  Notes
  =====
  The Savitzky-Golay is a type of low-pass filter, particularly suited
  for smoothing noisy data. The main idea behind this approach is to
  make for each point a least-square fit with a polynomial of high order
  over a odd-sized window centered at the point.

  Example
  =======

    >>> import criros
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = criros.filters.savitzky_golay(y, window_size=31, order=4)
    >>> plt.plot(t, y, label='Noisy signal')
    >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.show()

  @type  y: array_like, shape (N,)
  @param y: the values of the time history of the signal.
  @type  window_size: int
  @param window_size: the length of the window. Must be an odd integer number.
  @type  order: int
  @param order: the order of the polynomial used in the filtering. Must be less than C{window_size} - 1.
  @type  deriv: int
  @param deriv: the order of the derivative to compute (default = 0 means only smoothing)
  @rtype: ndarray, shape (N)
  @return: the smoothed signal (or it's n-th derivative).
  """
  import numpy as np
  from math import factorial

  try:
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
  except ValueError as msg:
    raise ValueError("window_size and order have to be of type int")
  if window_size % 2 != 1 or window_size < 1:
    raise TypeError("window_size size must be a positive odd number")
  if window_size < order + 2:
    raise TypeError("window_size is too small for the polynomials order")
  order_range = list(range(order+1))
  half_window = (window_size -1) // 2
  # precompute coefficients
  b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
  m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
  # pad the signal at the extremes with
  # values taken from the signal itself
  firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
  lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
  y = np.concatenate((firstvals, y, lastvals))
  return np.convolve( m[::-1], y, mode='valid')

def smooth_diff(n):
  """
  A smoothed differentiation filter (digital differentiator).

  Such a filter has the following advantages:

  First, the filter involves both the smoothing operation and
  differentiation operation. It can be regarded as a low-pass
  differentiation filter (digital differentiator).
  It is well known that the common differentiation operation amplifies
  the high-frequency noises. Therefore, the smoothed differentiation
  filter would be valuable in experimental (noisy) data processing.

  Secondly, the filter coefficients are all convenient integers (simple
  units) except for an integer scaling factor, as may be especially
  significant in some applications such as those in some single-chip
  microcomputers or digital signal processors.

  Example
  =======

    >>> import math
    >>> import criros
    >>> import scipy.signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> fs = 1. # Hz
    >>> samples = 250*fs
    >>> t = np.linspace(0, 3. / fs, num=samples)
    >>> noise = np.random.normal(0, 0.05, t.shape)
    >>> pos = np.sin(2*np.pi*fs*t)
    >>> vel = np.diff(pos+noise) * fs
    >>> smooth_window = 21
    >>> b = -criros.filters.smooth_diff(smooth_window)
    >>> vel_smooth = scipy.signal.lfilter(b, 1., pos+noise) * fs
    >>> plt.figure(), plt.subplot(211), plt.grid(True)
    >>> plt.plot(t, pos, 'k', lw=2.0, label='Original signal')
    >>> plt.plot(t, pos+noise, 'r', label='Noisy signal')
    >>> plt.legend(loc='best')
    >>> plt.ylabel(r'Position')
    >>> plt.subplot(212), plt.grid(True)
    >>> plt.plot(t[1:], vel, label='np.diff')
    >>> left_off = int(math.ceil(smooth_window / 2.0))
    >>> right_off = int(smooth_window - left_off)
    >>> plt.plot(t[left_off:-right_off], vel_smooth[smooth_window:], lw=2.0, label='smooth_diff')
    >>> plt.legend(loc='best')
    >>> plt.ylabel(r'Velocity')
    >>> plt.show()

  References
  ==========
    1.  Usui, S.; Amidror, I., I{Digital Low-Pass Differentiation for
        Biological Signal-Processing}. IEEE Transactions on Biomedical
        Engineering 1982, 29, (10), 686-693.
    2.  Luo, J. W.; Bai, J.; He, P.; Ying, K., I{Axial strain calculation
        using a low-pass digital differentiator in ultrasound
        elastography}. IEEE Transactions on Ultrasonics Ferroelectrics
        and Frequency Control 2004, 51, (9), 1119-1127.
  @type  n: int
  @param n: filter length (positive integer larger no less than 2)
  @rtype: ndarray, shape (n)
  @return: filter coefficients (anti-symmetry)
  """
  if n >= 2 and math.floor(n) == math.ceil(n):
    if n % 2 == 1:                    # is odd
      m = math.trunc((n-1) / 2.0);
      h = np.concatenate( (-np.ones(m), [0], np.ones(m)) ) / (m * (m+1))
    else:                             # is even
      m = math.trunc(n / 2.0);
      h = np.concatenate( (-np.ones(m), np.ones(m)) ) /m**2
  else:
    raise TypeError("The input parameter (n) should be a positive integer larger no less than 2.")
  return h

class ButterLowPass:
  """
  Butterworth lowpass digital filter design.

  Check C{scipy.signal.butter} for further details.
  """
  def __init__( self, cutoff, fs, order=5 ):
    """
    C{ButterLowPass} constructor

    @type  cutoff: float
    @param cutoff: Cut-off frequency in Hz
    @type  fs: float
    @param fs: The sampling frequency (Hz) of the signal to be filtered
    @type  order: int
    @param order: The order of the filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    self.b, self.a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)

  def __call__( self, x ):
    """
    Filters the input array across its C{axis=0} (each column is
    considered as an independent signal). Uses initial conditions (C{zi})
    for the filter delays.

    @type x: array
    @param x: An N-dimensional input array.
    @rtype: array
    @return: The output of the digital filter.
    """
    if not hasattr(self, 'zi'):
      cols = x.shape[1]
      zi = scipy.signal.lfiltic( self.b, self.a, [] ).tolist() * cols
      self.zi = np.array(scipy.signal.lfiltic( self.b, self.a, [] ).tolist() * cols)
      self.zi.shape = (-1, cols)
    (filtered, self.zi) = scipy.signal.lfilter(self.b, self.a, x, zi=self.zi, axis=0 )
    return filtered


class LowPassFO:
  """
  Causal implementation of a 1st-order Lowpass digital filter.
  """
  def __init__( self, cutoff, fs):
    """
    C{LowPassFO} constructor

    @type  cutoff: float
    @param cutoff: Cut-off frequency in Hz
    @type  fs: float
    @param fs: The sampling frequency (Hz) of the signal to be filtered
    @type  order: int
    @param order: The order of the filter.
    """
    self.b, self.a = lowpass_fo(cutoff, fs)

  def __call__( self, x ):
    """
    Filters the input array across its C{axis=0} (each column is
    considered as an independent signal). Uses initial conditions (C{zi})
    for the filter delays.

    @type x: array
    @param x: An N-dimensional input array.
    @rtype: array
    @return: The output of the digital filter.
    """
    if not hasattr(self, 'zi'):
      cols = x.shape[1]
      zi = scipy.signal.lfiltic( self.b, self.a, [] ).tolist() * cols
      self.zi = np.array(scipy.signal.lfiltic( self.b, self.a, [] ).tolist() * cols)
      self.zi.shape = (-1, cols)
    (filtered, self.zi) = scipy.signal.lfilter(self.b, self.a, x, zi=self.zi, axis=0 )
    return filtered
