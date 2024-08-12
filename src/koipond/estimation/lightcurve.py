import numpy as np 
from koipond.util.fun import find_biggest_gap, get_period, timeprint
import koipond.util.fun as util
from scipy.interpolate import interp1d
import seaborn as sn
import matplotlib.pyplot as plt
import os, scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import lightkurve as lk 
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_prominences

class KoiLightCurve:
    def __init__(self, lc_collection:lk.LightCurveCollection=None, kepid=''):
        self.lc_collection = lc_collection
        self.kepid = kepid

        # To be completled in #stitch method
        self.lightcurve = None
        self.raw_times = None
        self.raw_flux = None
        self.interpolating_x = None
        self.interpold_flux = None

        self.fold_lightcurves = []

        if not os.path.exists("debug"):
            os.mkdir("debug")
        if not os.path.exists(f"debug/{kepid}"):
            os.mkdir(f"debug/{kepid}")
        self.debug_folder = f"debug/{kepid}"

    def stitch(self, fn=None, sigma_upper=10, sigma_lower=float('inf'), debug=False):
        if fn is None:
            stitched = self.lc_collection.stitch().remove_nans()
        else:
            stitched = self.lc_collection.stitch(fn).remove_nans()
        stitched = stitched.remove_outliers(sigma_upper=sigma_upper, sigma_lower=sigma_lower)
        if debug:
            ax = stitched.scatter(title=f"kepid{self.kepid} stitched")
            ax.get_figure().savefig(f"{self.debug_folder}/stitched.png")
            plt.close(ax.get_figure())        

        self.lightcurve = stitched 

        raw_times = stitched['time'].value
        raw_flux = stitched['flux'].value
        to_sort = list(zip(raw_times, raw_flux))
        to_sort = sorted(to_sort, key=lambda x:x[0])
        raw_times, raw_flux = zip(*to_sort)
        self.raw_times = raw_times
        self.raw_flux = raw_flux

        period = get_period(raw_times)
        interpolating_x = np.arange(np.ceil(raw_times[0]), raw_times[-1], period)
        f = interp1d(x=raw_times, y=raw_flux, kind='linear')
        interpold_flux = f(interpolating_x)

        if debug:
            fig, ax = plt.subplots()
            ax.plot(interpolating_x, interpold_flux, 'k.', markersize=2)
            ax.set_title(f"Interpolated vals {self.kepid}")
            fig.savefig(f"{self.debug_folder}/inerpold-{self.kepid}.png")
            plt.close(fig)
        
        self.interpolating_x = interpolating_x
        self.interpold_flux = interpold_flux

    def locate_possible_tps(self, debug=False):
        freq, trans = self._perform_fft(removezerofreq=True, max_freq=2, min_freq=1./21, debug=debug)
        peaks, _ = find_peaks(trans, height=np.nanmean(trans))
        prominences, _, _ = peak_prominences(trans,peaks)
        sorted_prominences = list(sorted(prominences, reverse=True))
        prominence_threshold = sorted_prominences[min(len(sorted_prominences)//10, 30)]
        best_prominences_indexes = np.where(prominences>prominence_threshold)
        best_peaks = peaks[best_prominences_indexes]

        if debug:
            fig, ax = plt.subplots()
            ax.bar(freq, trans, width=0.01)
            ax.plot(best_peaks*get_period(freq), trans[best_peaks], 'rx', markersize=2)
            ax.set_title(f"Top {len(best_peaks)} peaks | kepid{self.kepid}")
            fig.savefig(f"{self.debug_folder}/top-peaks-kepid{self.kepid}.png")
            plt.close(fig)

        for peak in best_peaks:
            peak_frequency = freq[peak]
            detected_tp = 1/peak_frequency
            folding_lc = FoldingLightCurve(times=self.raw_times, fluxes=self.raw_flux, curve_id=f"{self.kepid}_{detected_tp}",
                                           estimated_tp=detected_tp)
            self.fold_lightcurves.append(folding_lc)

        if debug:
            with open(f"{self.debug_folder}/fft.log", 'w+') as log_file:
                log_file.write(f"Located {len(self.fold_lightcurves)} possible frequencies;\n")
                for fold_lc in self.fold_lightcurves:
                    log_file.write(f"Fold LC id {fold_lc.koi_id} | {fold_lc.estimated_tp}\n")

    def find_time_periods(self, iterations=10, debug=False):
        # good_detections = []
        for fold_lc in self.fold_lightcurves:
            if debug:
                timeprint(self.kepid, ": Estimating TP for folding LC with initial TP", fold_lc.estimated_tp)
            has_detected, estimated_tp = fold_lc.find_time_period(iterations=iterations, debug=debug)
            if has_detected:
                # good_detections.append(fold_lc)
                timeprint(f"Has detected planet with estimated tp {estimated_tp}.")
                return fold_lc
        if debug:
            timeprint("Done. No planet detected.")
        # if len(good_detections) > 0:
        #     good_detections = list(sorted(good_detections, key=lambda lc: lc.regression_score, reverse=True))
        #     return good_detections[0]
        # else:
        return None

    def _perform_fft(self, removezerofreq=False, max_freq=None, min_freq=None, debug=False):
        x = self.interpolating_x
        y = self.interpold_flux

        T = get_period(x)
        N = len(x)
        frequencies = fftfreq(N, T)
        # second half of freqs is -ve. We can ignore this as it is symmetrical
        frequencies = frequencies[:N//2]

        transformed = fft(y)
        # take only first half as is symmetrical for -ve numbers and we can ignore.
        transformed = np.abs(transformed[:N//2])

        if removezerofreq and frequencies[0] == 0.0:
            transformed[0] = 0

        # allows to remove samples of irrelevant lower frequencies for further calculations
        if max_freq is not None:
            remove_index = np.where(np.array(frequencies) >= max_freq)[0][0]
            frequencies = frequencies[:remove_index]
            transformed = transformed[:remove_index]

        if min_freq is not None:
            remove_index = np.where(np.array(frequencies) <= min_freq)[0][-1]
            frequencies = np.concatenate(([0]*remove_index, frequencies[remove_index:]))
            transformed = np.concatenate(([0]*remove_index, transformed[remove_index:]))
        
        if debug:
            width = (frequencies[-1] - frequencies[0]) / len(frequencies)
            fig, ax = plt.subplots()
            ax.bar(frequencies, 2.0 / N * transformed, width=width)
            ax.set_title(f"FFT Frequencies for {self.kepid}")
            fig.savefig(f"{self.debug_folder}/FFT Freqs {self.kepid}.png")
            plt.close(fig)
        
        return (frequencies, transformed)


class FoldingLightCurve:

    def __init__(self, times=[], fluxes=[], curve_id='', estimated_tp=1):
        self.times = times
        self.fluxes = fluxes
        self.koi_id = curve_id
        self.estimated_tp = estimated_tp
        self.iteration_count = 0
        self.has_detected_planet = False

        self.regression_score = -1 # if -1: no estimation iteration has happened

        if not os.path.exists("debug"):
            os.mkdir("debug")
        repeat_count = 0
        while os.path.exists(f"debug/{curve_id}-{repeat_count}"):
            repeat_count += 1
        self.debug_folder = f"debug/{curve_id}-{repeat_count}"
        os.mkdir(self.debug_folder)

    def find_time_period(self, iterations=10, debug=False):
        for i in range(iterations):
            smoothed_img, folded_dict, interp_dict = self.get_smoothed_img(debug=debug)
            if smoothed_img is None:
                if debug:
                    timeprint(f"{self.koi_id} Fold was bad on tp {self.estimated_tp}; skipping")
                return False, self.estimated_tp
            raw_img = np.array([np.log(x[1]+0.00001) for x in interp_dict.values()]) 
            new_data = self.find_multiples(smoothed_img, raw_img,debug=debug)
            if new_data is not None:
                smoothed_img, folded_dict, interp_dict = new_data
            if smoothed_img is None:
                if debug:
                    timeprint(f"{self.koi_id} Fold was bad on tp {self.estimated_tp}; skipping")
                return False, self.estimated_tp
            has_corr, has_low_grad = self._estimate_tp(smoothed_img, folded_dict, interp_dict, debug=debug)
            if not has_corr:
                # returning false = no solid correlation found in this value; continue to next TP.
                return False, self.estimated_tp
            elif has_low_grad:
                timeprint("HAS LOW GRADIENT")
                # self.find_multiples(smoothed_img,debug=debug)
        if not self.has_detected_planet:
            timeprint("Checking for transit confirmation.")
            smoothed_img, _,_ = self.get_smoothed_img(debug=debug)
            if smoothed_img is None:
                if debug:
                    timeprint(f"{self.koi_id} Fold was bad on tp {self.estimated_tp}; skipping")
                return False, self.estimated_tp
            if self.verify_transits(smoothed_img, debug=debug):
                self.has_detected_planet = True
                timeprint("Has detected planet.")
        return self.has_detected_planet, self.estimated_tp

    def get_smoothed_img(self,debug=False):
        if self.estimated_tp < 0:
            timeprint(f"WARNING: Estimated TP < 0; skipping this FoldedLightCurve estimation;")
            return False
        if self.estimated_tp > np.max(self.times) - np.min(self.times):
            timeprint(f"WARNING: Calculated TP outside of bounds of given light curve; skipping.")
            return False

        if debug:
            timeprint(f"{self.koi_id} : Estimating TP; iteration {self.iteration_count}: initial estimation {self.estimated_tp}")

        interp_size = 1200
        epoch_time = np.min(self.times)
        
        folded_dict, interp_dict = self._fold(tp=self.estimated_tp, epoch_time=epoch_time, normalise='median-norm', transform=None, interp_size=interp_size, debug=debug)

        if len(folded_dict) == 0:
            return None, None, None

        # get interp'd flux values for each fold, log, and stick into 2D array, creating an "image".
        # adding a very small val to each row before logging: normalised in folding means guaranteed 0-val (bad in logs)
        folded_img = np.array([np.log(x[1]+0.00001) for x in interp_dict.values()])
        smoothed_img = scipy.ndimage.gaussian_filter(folded_img, sigma=(2,10))

        if debug:
            timeprint("Generated image.")
            fig, ax = plt.subplots()
            for (x,y) in folded_dict.values():
                ax.plot(x,y,'k.',markersize=2)
            ax.set_title(f"Folded iter {self.iteration_count} estimated tp {self.estimated_tp}")
            fig.savefig(f"{self.debug_folder}/iteration{self.iteration_count} tp {self.estimated_tp} Folded curve.png")
            plt.close(fig)

            sn.heatmap(folded_img, annot=False)
            plt.title(f"estimated_tp={self.estimated_tp} iteration {self.iteration_count}")
            plt.savefig(f"{self.debug_folder}/folded-{self.iteration_count}-{self.estimated_tp}.png")
            plt.close()
    
            timeprint(f"Smoothed image.")
            sn.heatmap(smoothed_img, annot=False)
            plt.title(f"SMOOTHED sigma=(2,10) estimated_tp={self.estimated_tp} iteration {self.iteration_count}")
            plt.savefig(f"{self.debug_folder}/folded-smoothed-{self.iteration_count}-{self.estimated_tp}.png")
            plt.close()

        #TODO: this might need to turn to a variable sigma? seemed to work well on kepler-229c.
        return smoothed_img, folded_dict, interp_dict

    def find_multiples(self,smoothed_img,raw_img,debug=False):
        timeprint(f"Checking for multiples of tp {self.estimated_tp}")
        histogram = np.sum(smoothed_img,axis=0) / len(smoothed_img)
        transit_count = self.count_transits(histogram,debug=debug)
        zero_vals = np.where(np.abs(histogram)<1)[0]
        detected = transit_count >= 1 and len(histogram[zero_vals])>len(histogram)//2
        if detected:
            if transit_count > 1:
                self.estimated_tp /= transit_count
                smoothed_img, folded_dict, interp_dict = self.get_smoothed_img(debug=debug)
                if smoothed_img is None:
                    return None
                raw_img = np.array([np.log(x[1]+0.00001) for x in interp_dict.values()]) 
                data = smoothed_img, folded_dict, interp_dict
            densities = [self.count_transits(scipy.ndimage.gaussian_filter1d(row,sigma=10),mode='verify',debug=False) for row in raw_img]
            transit_density = sum(densities)/len(densities)
            if debug:
                timeprint("Transit density", transit_density)
            if transit_density < 0.65 and transit_density > 0:
                to_multiply = round(1/transit_density)
                self.estimated_tp *= to_multiply
                if debug:
                    timeprint(f"Found low transit density. Refactoring by {to_multiply}. New tp: {self.estimated_tp}")
                data = self.get_smoothed_img(debug=debug)
            elif transit_count == 1:
                data = self.get_smoothed_img(debug=debug)
            if debug:
                timeprint(f"Found {transit_count} multiples on current. New tp: {self.estimated_tp}")
        else:
            data = None
            if debug:
                timeprint(f"No good transits found.")
        if data is not None and data[0] is None:
            data = None
        return data

    def verify_transits(self, smoothed_img, debug=False):
        histogram = np.sum(smoothed_img,axis=0) / len(smoothed_img)
        transit_count = self.count_transits(histogram, mode='verify', debug=debug)
        zero_vals = np.where(np.abs(histogram)<1)[0]
        return transit_count == 1 and len(histogram[zero_vals])>len(histogram)//2

    def count_transits(self, histogram, mode='lap', debug=False):
        histogram = scipy.ndimage.gaussian_filter1d(histogram,sigma=10)
        sobel_hist = scipy.ndimage.sobel(histogram)
        lap_hist = scipy.ndimage.laplace(histogram)
        crossed = util.zero_cross(sobel_hist)
        lap_threshold = np.std(lap_hist)**2

        threshold_count = len(histogram)//10
        threshold = sorted(histogram.flatten())[threshold_count]

        # check that only one transit dip detected per transit (curve may be wonkey, only count one if not gone back up to 0 first.)
        if mode == 'verify':
            detected_transit_indexes = np.where(crossed & (histogram<threshold))[0]
        else:
            detected_transit_indexes = np.where(crossed & (lap_hist>lap_threshold) & (histogram<threshold))[0]
        detected_transits = {}
        zero_threshold = np.mean(np.abs(histogram)) + np.std(histogram)
        zero_vals = np.where(np.abs(histogram)<zero_threshold)[0]
        for transit_index in detected_transit_indexes:
            transit_min, transit_max = zero_vals[np.where(zero_vals<=transit_index)[0]],zero_vals[np.where(zero_vals>transit_index)[0]]
            transit_min, transit_max = -1 if len(transit_min) == 0 else transit_min[-1], -1 if len(transit_max) == 0 else transit_max[0]
            arr = detected_transits.get((transit_min,transit_max),[])
            arr.append(transit_index)
            detected_transits.update({(transit_min,transit_max):arr})
        actual_transits = []
        for val in detected_transits.values():
            actual_transits.append(np.median(val))
        transit_count = len(actual_transits)
        if debug:
            timeprint(f"Prop of low histogram vals is {len(histogram[np.where(np.abs(histogram)<0.5)])/(len(histogram)//2)}")
            fig, ax = plt.subplots()
            hist_x = np.linspace(0,1,len(histogram))
            ax.plot(hist_x,histogram,'k.',markersize=2, label="Histogram")
            crossed = np.where(crossed & (lap_hist>lap_threshold))[0]
            ax.plot(hist_x[crossed], np.zeros(len(crossed)),'rx',markersize=3)
            ax.plot(np.linspace(0,1,len(sobel_hist)), sobel_hist, 'g.',markersize=2, label="First Order")
            ax.plot(np.linspace(0,1,len(lap_hist)), lap_hist, 'b.',markersize=2, label="Second Order")
            ax.set_title(f"Detected Transits ({transit_count})")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            fig.savefig(f"{self.debug_folder}/{self.iteration_count}_{self.koi_id}_detected_transits.png")
            plt.close(fig)
        return transit_count


    def _estimate_tp(self, smoothed_img, folded_dict, interp_dict, debug=False):
        timeprint(f"{self.koi_id} : Performing gradient descent or summin on init TP {self.estimated_tp}")
        self.iteration_count += 1
        
        # TODO threshold might need to be variable? re-adress this later on. likely will need to change.
        allowed_points_per_fold = 30
        threshold_index = len(interp_dict) * allowed_points_per_fold
        threshold = sorted(smoothed_img.flatten())[threshold_index]

        # threshold = -0.2 
        line_filter = smoothed_img<threshold

        rows, cols = np.where(line_filter)
        cols *= -1
        rows, cols = rows.reshape(-1,1), cols.reshape(-1,1)

        # has_corr = self.verify_hough(smoothed_img,line_length=(len(folded_dict)//6)*5,
        #                                         line_gap=10, signifance_thresh=(len(folded_dict)/3)*2,
        #                                         debug=debug)
        # TODO: either remove hough verification or fix it; temporarily removing right now.
        has_corr = True   

        interp_size = 1200

        if has_corr:
            # gradient, grad_dev = self.svr_gradient(rows,cols,eps=np.sqrt(len(folded_dict))*5/6,stack_lines=lambda x:len(x)>40 and np.average([len(a) for a in x.values()])<100,debug=debug)
            gradient, grad_dev = self.svr_gradient(rows,cols,eps=np.sqrt(len(folded_dict))*0.9,stack_lines=lambda x:len(x)>40 and np.average([len(a) for a in x.values()])<100,debug=debug)

            if gradient is None:
                # no point in continuing this; clustering is bad therefore no need to continue searching
                return False, False

            reformatting_factor = self.estimated_tp/interp_size
            learning_constant = 0.15
            learning_rate = min(1.0,0.5+(1/np.exp(learning_constant*self.iteration_count)))
            new_estimated_tp = self.estimated_tp + (gradient * reformatting_factor * learning_rate)

            grad_threshold = (interp_size/5)/len(folded_dict)
            has_low_gradient = gradient<grad_threshold

        if debug:
            fig, ax = plt.subplots()
            max_x = -100
            min_x = 100000000
            for (x,y) in interp_dict.values():
                ax.plot(x,y,'k.',markersize=2)
                if min(x) < min_x:
                    min_x = min(x)
                if max(x) > max_x:
                    max_x = max(x)
            # temp_threshold = sorted(np.array([x[1] for x in interp_dict.values()]).flatten())[threshold_index]
            ax.plot([min_x,max_x],[threshold,threshold],'g-')
            ax.set_title(f"{self.koi_id}, iter {self.iteration_count} : Threshold on values for SVM")
            fig.savefig(f"{self.debug_folder}/{self.iteration_count}_{self.koi_id} threshold.png")
            plt.close(fig)

            with open(f"{self.debug_folder}/iteration-{self.iteration_count}.log", 'w+') as log_file:
                log_file.write(f"ITERATION {self.iteration_count}\n")
                log_file.write(f"Original estimated tp = {self.estimated_tp}\n")
                log_file.write(f"Folding with interpolation count {interp_size}, normalisation median-norm, no transform\n")
                log_file.write(f"Logged folded img\n")
                log_file.write("Smoothed folded img with sigma=(2,10)\n")
                # log_file.write(f"Line threshold = (num_folds * {allowed_points_per_fold})th lowest value in normalised fluxes [{threshold}]\n")
                log_file.write(f"Stacking folded lines using DBSCAN, eps={np.sqrt(len(folded_dict)+10)}\n")
                log_file.write(f"Cut stacked line to +/-5% of midpoint for each row.\n")
                log_file.write(f"Using hough transforms to verify existence of correlation with significance threshold = {(len(folded_dict)//3)*2}\n")
                if has_corr:
                    log_file.write(f"Using SVM to find linear regression with std dev in gradients = {grad_dev}.\n")
                    log_file.write(f"Regression score {self.regression_score}\n")
                    log_file.write(f"Calculated gradient {gradient}\n")
                    log_file.write(f"Reformatting factor {reformatting_factor}\n")
                    log_file.write(f"Learning rate {learning_rate} (1/exp({learning_constant}*iter))\n")
                    log_file.write(f"Adjusting estimated tp by {gradient*reformatting_factor*learning_rate}\n")
                    log_file.write(f"New estimated tp: {new_estimated_tp}\n")
                else:
                    log_file.write(f"No correlation found; no gradient adjustments made; no further actions needed for estimated tp {self.estimated_tp}\n")

        if has_corr:
            self.estimated_tp = new_estimated_tp
        if debug:
            timeprint(f"{self.koi_id} : Estimated TP; iteration {self.iteration_count}: new estimation {self.estimated_tp}")
        return has_corr, has_low_gradient


    def svr_gradient(self,rows,cols,eps,stack_lines=lambda x:False,debug=False):
        if debug:
            timeprint(f"Applying SVM to {len(cols)} found data.")

            fig,ax = plt.subplots()
            ax.plot(rows,cols,'k.',markersize=2)
            ax.set_title(f"Size={len(cols)}")
            fig.savefig(f"{self.debug_folder}/{self.iteration_count}_{self.koi_id}_BINARY_IMG.png",bbox_inches='tight')
            plt.close(fig)
        
        if len(cols) == 0:
            timeprint("No data available to apply to SVM. Skipping.")
            return None,None

        centroids, clusters = self.dbscan(rows,cols,eps=eps,debug=debug)
        if stack_lines(clusters):
            rows, cols = self.stack(centroids, clusters, debug=debug)
            rows, cols = self.reduce(clusters={1:zip(rows,cols)}, fraction_to_keep=0.001, debug=debug)
            if len(rows) == 0:
                return None, 0

            std_dev = self.std_dev(rows.ravel(), cols.ravel(), debug=debug)
            if debug:
                timeprint(f"Using espilon=std dev of {std_dev} on SVM.")

            x_train, x_test, y_train, y_test = train_test_split(rows, cols, test_size=0.3)
            x_train -= x_train[0]
            x_test -= x_test[0]
            regression = make_pipeline(RandomizedSearchCV(LinearSVR(epsilon=std_dev, dual="auto")))
                                                                            # param_distributions={'C':scipy.stats.reciprocal(0.1,100)}))
            regression.fit(x_train,np.ravel(y_train))
            
            self.regression_score = regression.score(x_test,np.ravel(y_test))

            y_pred = regression.predict(sorted(x_test))

            gradient = (y_pred[0]-y_pred[-1])/(sorted(x_test)[-1]-sorted(x_test)[0])[0]

            if debug:
                fig, ax = plt.subplots()
                ax.plot(rows.reshape(-1,1), cols.reshape(-1,1),'k.',markersize=1)
                ax.plot(sorted(x_test), y_pred,'g.')
                y_intercept = y_pred[0] - (-1*gradient*min(x_test))
                ax.plot(x_test.reshape(-1,1), y_intercept + ((-1*gradient)*x_test.reshape(-1,1)))
                ax.set_title(f"SVM estimated_tp={self.estimated_tp} iteration {self.iteration_count} score {self.regression_score}")
                fig.savefig(f"{self.debug_folder}/SVM-{self.iteration_count}-{self.estimated_tp}-reg-score={self.regression_score}.png")
                plt.close(fig)
        else:
            all_grads = []
            weights = []

            cmap = plt.cm.get_cmap('rainbow')
            colours = [cmap(i/40) for i in range(40)]
            colour_count = 0

            if debug:
                fig, ax = plt.subplots()
            for label, cluster in clusters.items(): 
                cluster_rows, cluster_cols = zip(*cluster)
                cluster_rows, cluster_cols = self.reduce(clusters={label:cluster}, fraction_to_keep=0.5 if len(clusters)>1 else 0.001, debug=debug)
                std_dev = self.std_dev(cluster_rows.ravel(), cluster_cols.ravel(), debug=debug)
                if len(cluster_rows) == 0:
                    continue
                if debug:
                    timeprint(f"Finding SVM on cluster {label} with std dev {std_dev/2}")
                x_train, x_test, y_train, y_test = train_test_split(cluster_rows, cluster_cols, test_size=0.2)
                regression = make_pipeline(StandardScaler(), LinearSVR(epsilon=std_dev/2, dual="auto"))
                regression.fit(x_train,np.ravel(y_train))
                self.regression_score = regression.score(x_test,np.ravel(y_test))
                y_pred = regression.predict(sorted(x_test))
                gradient = (y_pred[0]-y_pred[-1])/(sorted(x_test)[-1]-sorted(x_test)[0])[0]
                
                if not np.isnan(gradient):
                    all_grads.append(gradient)
                    weights.append(len(cluster_rows))

                if debug:
                    ax.plot(cluster_rows, cluster_cols, '.', color=colours[colour_count%len(colours)],markersize=2,label=str(label))
                    ax.plot(cluster_rows, cluster_cols, '.', color='k',markersize=1,label=str(std_dev))
                    ax.plot(x_test.reshape(-1,1), ((-1*gradient)*(x_test.reshape(-1,1)-x_test[0]))+y_test[0], 'r-')
                    colour_count += 1


            gradient = np.average(all_grads,weights=weights)
            std_dev = np.std(all_grads)

            if debug:
                timeprint(f"Calculated {len(all_grads)} gradients: {all_grads}")
                
                y_intercept = np.median(cols.flatten()) - (-1*gradient*min(rows.flatten()))
                ax.plot(rows.reshape(-1,1), y_intercept + ((-1*gradient)*rows.reshape(-1,1)))
                ax.set_title(f"SVM estimated_tp={self.estimated_tp} iteration {self.iteration_count} score {self.regression_score}")
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                fig.savefig(f"{self.debug_folder}/SVM-{self.iteration_count}-{self.estimated_tp}-reg-score={self.regression_score}.png")
                plt.close(fig)



        return gradient, std_dev
    
    def verify_hough(self, img, line_length, line_gap, signifance_thresh, debug=False):
        edges = canny(img)
        lines = probabilistic_hough_line(edges, line_length=line_length,
                                        line_gap=line_gap)
        
        if debug:
            timeprint("Verifying hough lines.")
            fig, ax = plt.subplots()

        num_accepted = 0
        for p0,p1 in lines:
            vertical_height = abs(p0[0]-p1[0])
            if debug:
                ax.plot((p0[1], p1[1]), (p0[0], p1[0]), color='k' if vertical_height<signifance_thresh else 'g')
            if vertical_height < signifance_thresh:
                continue
            if not debug:
                return True
            else:
                num_accepted += 1 

        if debug:
            ax.set_title(f"iter {self.iteration_count} hough lines tp {self.estimated_tp}")
            fig.savefig(f"{self.debug_folder}/{self.iteration_count}_{self.estimated_tp}_hough_lines.png")
            plt.close(fig)
            timeprint(f"Found {num_accepted} hough lines above given thrshold (eliminated {len(lines)-num_accepted})")

        return num_accepted > 0

    def dbscan(self, rows, cols, eps=5, debug=False):
        if debug:
            timeprint(f"Applying DBSCAN to {len(cols)} data points.")
        zipped_data = np.array(list((zip(rows.ravel(),cols.ravel()))))
        if (len(zipped_data) > 50_000):
            centroid = {1:(np.mean(zipped_data[:,0]), np.mean(zipped_data[:,1]))}
            cluster = {1:zipped_data}
            return centroid, cluster
        db = DBSCAN(eps=eps, min_samples=20).fit(X=zipped_data)
        labels = db.labels_

        if debug:
            fig, ax = plt.subplots()

        clusters = {}
        centroids = {}

        for k in set(labels):
            cluster = zipped_data[labels==k]
            if k>=0 and len(cluster) > 0:
                clusters.update({k:cluster})
                centroid = (np.mean(cluster[:,0]),np.mean(cluster[:,1]))
                centroids.update({k:centroid})
            if debug:
                ax.plot(cluster[:,0],cluster[:,1], '.', markersize=3,label=f"{k} s={len(cluster)}")
                ax.plot(centroid[0],centroid[1], 'x',markersize=3)

        if debug:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            fig.savefig(f"{self.debug_folder}/{self.iteration_count}_{self.koi_id}_DBSCAN_clusters.png",bbox_inches='tight')
            plt.close(fig)

        centroids = dict(sorted(centroids.items(), key=lambda x: x[1], reverse=True))
        return centroids, clusters

    def stack(self, centroids, clusters, debug=False):
        centroids = sorted([(label,centroid) for label,centroid in centroids.items()], key=lambda x:x[1][1])
        minmax = {}
        for label, cluster in clusters.items():
            minmax.update({label:(np.min(cluster[:,0]),np.max(cluster[:,0]))})

        new_coords = clusters.get(centroids[0][0])
        for i in range(1,len(centroids)):
            current_label, _ = centroids[i]
            current_min, _ = minmax.get(current_label)
            previous_label, _ = centroids[i-1]
            _, previous_max = minmax.get(previous_label)
            
            cluster = clusters.get(current_label)
            cluster = cluster + [0,(previous_max-current_min)]
            new_coords = np.concatenate((new_coords, cluster))
        
            clusters.update({current_label:cluster})
            minmax.update({current_label:(np.min(cluster[:,1]),np.max(cluster[:,1]))})
            centroids[i] = (label,np.mean(cluster[:,1]))

        straight_line_rows, straight_line_cols = list(zip(*new_coords))
        straight_line_rows = np.array(straight_line_rows)
        straight_line_cols = np.array(straight_line_cols)

        if debug:
            fig, ax = plt.subplots()
            ax.plot(straight_line_cols,straight_line_rows,'k.',markersize=2)
            fig.savefig(f"{self.debug_folder}/{self.iteration_count}_{self.koi_id}_stacked_and_reduced.png")
            plt.close(fig)

        return straight_line_rows.reshape(-1,1), straight_line_cols.reshape(-1,1)

    def reduce(self, clusters, fraction_to_keep=0.1, debug=False):
        new_line_vals = [] 

        for label, cluster in clusters.items():
            rows, cols = list(zip(*cluster))
            rows = np.array(rows).flatten()
            cols = np.array(cols).flatten()
            for row in set(rows):
                all_in_row = list(sorted([c[1] for c in cluster if c[0]==row]))
                if len(all_in_row) == 0:
                    continue
                midpoint = len(all_in_row)//2
                num_to_include = max(6,round(len(all_in_row)*fraction_to_keep))
                to_keep = all_in_row[midpoint-num_to_include//2:midpoint+num_to_include//2]
                for col in to_keep:
                    new_line_vals.append([row,col])

        if len(new_line_vals) == 0:
            return  [], []

        reduced_rows, reduced_cols = zip(*new_line_vals)
        reduced_rows = np.array(reduced_rows).reshape(-1,1)
        reduced_cols = np.array(reduced_cols).reshape(-1,1)
        return reduced_rows, reduced_cols

    def std_dev(self, rows, cols, debug=False):
        rows = rows.flatten()
        cols = cols.flatten()
        zipped = list(zip(rows,cols))
        rows_dict = {r:[c[1] for c in zipped if c[0]==r] for r in set(rows)}
        probably_flat_line = np.concatenate([np.array(c)-np.median(c) for c in rows_dict.values() if len(c)>0])
        std_dev = np.std(probably_flat_line) if len(probably_flat_line) > 0 else 0.5
        return std_dev

    def _fold(self, tp, epoch_time=0, normalise=None, transform=None, interp_size=1200, debug=False):
        times = np.copy(self.times)-epoch_time
        fold_count = 0
        interp_folded = {}
        folded_dict = {}
        
        flux = np.copy(self.fluxes)

        if normalise == 'norm':
            flux = (flux-np.min(flux))/(np.max(flux)-np.min(flux))
        elif normalise == 'mean-norm':
            flux = (flux-np.min(flux))/(np.mean(flux)-np.min(flux))
        elif normalise == 'median-norm':
            flux = (flux-np.min(flux))/(np.median(flux)-np.min(flux))

        if transform == 'log':
            flux = np.log(flux)

        skipped_count = 0
        interp_x = np.linspace(-tp/2, tp/2, interp_size)
        fill_val = np.median(flux)

        start_time =  - (((epoch_time-times[0])%tp)+1)*tp
        end_time = start_time + tp

        while start_time < times[-1]:
            start_time =times[0] + (fold_count * tp)
            end_time =times[0] + ((fold_count + 1) * tp)

            fold_count += 1

            start_index = np.where(times > start_time)[0]
            if len(start_index) == 0:
                start_time += tp
                end_time += tp
                continue
            start_index = start_index[0]

            end_index = np.where(times < end_time)[0]
            if len(end_index) == 0:
                start_time += tp
                end_time += tp
                continue
            end_index = end_index[-1]

            if start_index >= end_index:
                start_time += tp
                end_time += tp
                continue

            fold_times =times[start_index:end_index] - ((start_time+end_time)/2)
            fold_flux = flux[start_index:end_index]
            
            to_test = np.concatenate(([-tp/2], fold_times, [tp/2]))
            biggest_gap = find_biggest_gap(to_test)
            if biggest_gap > tp * 0.1:
                skipped_count += 1
                start_time += tp
                end_time += tp
                continue

            f = interp1d(x=fold_times, y=fold_flux, kind='linear', bounds_error=False, fill_value=fill_val)

            interpold = f(interp_x)
            interp_folded.update({str(fold_count):(interp_x, interpold)})

            folded_dict.update({str(fold_count):(fold_times,fold_flux)})

            start_time += tp
            end_time += tp

        if skipped_count > 0 and debug:
            timeprint(f"{self.koi_id} : Skipped {skipped_count} of {skipped_count + len(interp_folded)} folds due to gaps.")
        return folded_dict, interp_folded
