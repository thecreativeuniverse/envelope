from scipy.signal import decimate
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def timeprint(*toprint):
   print(f"[{datetime.now().strftime('%H:%M:%S')}]", *toprint)

def downsample(x=[], ys=[], q=1):
    decimated = decimate(ys, q)
    f = interp.interp1d(x=np.linspace(min(x), max(x), len(decimated[0])), y=decimated, kind='cubic')
    return f(x)

def find_biggest_gap(data):
  max_gap = -1
  for i in range(len(data)-1):
    gap = data[i+1]-data[i]
    if gap>max_gap:
      max_gap = gap
  return max_gap

def average(data):
    data = np.nan_to_num(data, nan=0)
    return sum(data) / len(data)

def get_period(arr):
    period = []
    for i in range(1, len(arr)):
        period.append(arr[i] - arr[i - 1])
    period = average(period)
    return period

def zero_cross(arr):
  crossings = []
  prev_sign = arr[0]
  for i in range(1,len(arr)):
    first = arr[i-1]
    second = arr[i]
    res = first*second
    crossings.append(res<0 or ((res == 0) and ((second>=0 and prev_sign<0) or (second<0 and prev_sign>=0))))
    prev_sign = first
  crossings.append(False)
  return crossings