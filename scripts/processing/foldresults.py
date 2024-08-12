import pandas as pd
import matplotlib.pyplot as plt 
import lightkurve as lk
import os
from koipond.estimation.lightcurve import FoldingLightCurve
import seaborn as sn

if __name__ == '__main__':
    df = pd.read_csv('data/results.csv')

    if not os.path.exists(os.path.join(os.getcwd(), "debug/results")):
        os.mkdir(f"debug/results")

    file_names = os.listdir(os.path.join(os.getcwd(), "debug/results"))
    
    for index, row in df.iterrows():
        print(f"Processing {index} of {len(df)} results.")
        kepid = int(row['kepid'])
        
        if f"{kepid}_result.png" in file_names:
            print(f"Already processed {kepid}; skipping.")
            continue

        koi_period = row['koi_period']
        envelope_period = row['envolope_period']
        koi_disposition = row['koi_disposition']

        lc_coll = None
        while lc_coll is None:        
            try:
                res = lk.search_lightcurve(f'KIC{kepid}', author=['Kepler', 'K2'], exptime='long')
                lc_coll = res.download_all()
            except Exception as e:
                lc_coll = None
        lc = lc_coll.stitch(lambda x: x.remove_nans().flatten().normalize()).remove_outliers(sigma_upper=10, sigma_lower=15)
        

        raw_times = lc['time'].value
        raw_flux = lc['flux'].value
        to_sort = list(zip(raw_times, raw_flux))
        to_sort = sorted(to_sort, key=lambda x:x[0])
        raw_times, raw_flux = zip(*to_sort)


        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(25,25))
        lc.scatter(ax=ax1)
        ax1.set_title(f"kepid {kepid} stitched")

        flc = FoldingLightCurve(times=raw_times, fluxes=raw_flux,curve_id=str(kepid), estimated_tp=koi_period)
        smoothed_img, folded_dict, interp_dict = flc.get_smoothed_img()
        if folded_dict is not None:
            for (x,y) in folded_dict.values():
                ax2.plot(x,y,'k.',markersize=2)
            ax2.set_title(f"KOI Period {koi_period}")
        else:
            ax2.set_title(f"Folded Dict None, KOI Period {koi_period}")
        ax2.set_ylabel("Normalised Relative Flux")
        ax2.set_xlabel("Phase")

        flc = FoldingLightCurve(times=raw_times, fluxes=raw_flux,curve_id=str(kepid), estimated_tp=envelope_period)
        smoothed_img, folded_dict, interp_dict = flc.get_smoothed_img()
        if smoothed_img is not None:
            sn.heatmap(smoothed_img, ax=ax3)
            ax3.set_title(f"ENVELOPE Period {envelope_period}")
        else:
            ax3.set_title(f"Smoothed img None, ENVELOPE Period {envelope_period}")
        if folded_dict is not None:
            for (x,y) in folded_dict.values():
                ax4.plot(x,y,'k.',markersize=2)
            ax4.set_title(f"ENVELOPE Period {envelope_period}")
        else:
            ax4.set_title(f"Foled Dict None, ENVELOPE Period {envelope_period}")
        
        fig.savefig(f"debug/results/{kepid}_result.png")
        plt.close(fig)