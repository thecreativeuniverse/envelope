import os

from lightkurve import LightCurve
import scipy.interpolate as interp
from pandas import DataFrame
import pandas
from koipond.util.constants import *
import torch

class InputLightCurve:

    def __init__(self, kid: str, lightcurve: LightCurve, label=None):
        self.kid = kid
        self._raw_lightcurve = lightcurve.flatten().remove_nans()
        self._has_prepared = False
        self.prepared_lightcurves = []
        self.label = label

    def get_gap(self, times):
        separations = []
        for i in range(len(times)-1):
            t1 = times[i]
            t2 = times[i+1]
            separations.append(t2-t1)
        return max(separations), separations

    def prepare(self):
        if self._has_prepared:
            raise RuntimeError("LightCurve already prepared")

        flattened_lc = self._raw_lightcurve.flatten()
        raw_times = flattened_lc['time'].value
        raw_fluxes = flattened_lc['flux'].value

        interpolating_x = np.arange(np.ceil(min(raw_times)), max(raw_times), DATA_SEPARATION)
        f = interp.interp1d(x=raw_times, y=raw_fluxes, kind='linear')
        interpolated_fluxes = f(interpolating_x)

        remainder = int(len(interpolating_x) % INPUT_SIZE)
        if remainder > 0:
            interpolating_x = interpolating_x[:-remainder]
            interpolated_fluxes = interpolated_fluxes[:-remainder]

        num_bins = int(len(interpolating_x) / INPUT_SIZE)
        for input_bin in range(1, num_bins + 1):
            start = int(INPUT_SIZE * (input_bin - 1))
            end = int(INPUT_SIZE * input_bin)
            raw_bin_times = raw_times[(raw_times > interpolating_x[start]) & (raw_times < interpolating_x[end-1])]
            time_data = interpolating_x[start:end]

            if len(raw_bin_times) == 0 or len(time_data) == 0:
              continue

            gap, separations = self.get_gap(raw_bin_times)

            acceptable_gap = 0.2

            if gap > acceptable_gap:
                try:
                    gap_start_index = np.where(separations == gap)[0][0]
                    gap_end_index = np.where(separations == gap)[0][0] + 1
                    new_start = start + gap_end_index
                    new_end = end + gap_end_index

                    time_data = interpolating_x[new_start:new_end]
                    new_raw_bin_times = raw_times[(raw_times > interpolating_x[new_start]) & (raw_times < interpolating_x[new_end-1])]
                    new_gap, _ = self.get_gap(new_raw_bin_times)
                except IndexError as e:
                    new_gap = 1.0 # to ensure next section is executed
                    pass
                if(new_gap > acceptable_gap):
                    try:
                        new_start = start - gap_start_index
                        new_end = end - gap_start_index
                        time_data = interpolating_x[new_start:new_end]
                        new_raw_bin_times = raw_times[(raw_times > interpolating_x[new_start]) & (raw_times < interpolating_x[new_end-1])]
                        new_gap, _ = self.get_gap(new_raw_bin_times)
                        if new_gap > acceptable_gap:
                            raise RuntimeError("Could not shift gap")
                    except Exception as e:
                        print(e)
                        continue
                
                start, end = new_start, new_end

            if len(time_data) != INPUT_SIZE:
                raise RuntimeError(f"Something wrong with your code. Time data should be {INPUT_SIZE} but is actually {len(time_data)}.")

            data = BinnedLightCurve(self.kid + "_" + str(input_bin),
                                    time_data, interpolated_fluxes[start:end])
            self.prepared_lightcurves.append(data)

        self._has_prepared = True

    def save_data(self, dir_name):
        if not self._has_prepared:
            raise RuntimeError("Lightcurve not yet prepared")
        if not os.path.exists(os.path.join(os.getcwd(), dir_name)):
            os.makedirs(dir_name)
        if self.label is not None:
            if not os.path.exists(os.path.join(os.getcwd(), dir_name, "init/labels.csv")):
              label_df = DataFrame({'curve_id':[],'label':[]})
            else:
              label_df = pandas.read_csv(filepath_or_buffer=f"{dir_name}/init/labels.csv")
        else:
            label_df = None
        print("num blcs", len(self.prepared_lightcurves), len(set([blc.curve_id for blc in self.prepared_lightcurves])), "individuals") #debugging
        for blc in self.prepared_lightcurves:
            blc.to_pt(dir_name)
            print(f"saving {blc.curve_id} as pt") #debugging
            if label_df is not None:
                print(f"saving {blc.curve_id} to labels.csv") #debugging
                label_df = pandas.concat((label_df, DataFrame({'curve_id':[blc.curve_id],'label':[self.label]})), ignore_index=True)
        if label_df is not None:
            print("saving label_Df to file") #debugging
            label_df.to_csv(path_or_buf=f"{dir_name}/init/labels.csv", index=False)

class BinnedLightCurve:
    def __init__(self, curve_id, times, flux):
        self.curve_id = curve_id
        self.times = times
        self.flux = flux

    def to_csv(self, dir_name):
        df = DataFrame({'time': self.times, 'flux': self.flux})
        df.to_csv(path_or_buf=f"{dir_name}/{self.curve_id}.csv")

    def to_pt(self, dir_name):
        tensor = torch.Tensor(self.flux)
        torch.save(tensor, f=f"{dir_name}/{self.curve_id}.pt")

def get_classification_label(disposition):
    return 1 if disposition == 'CONFIRMED' else 0
