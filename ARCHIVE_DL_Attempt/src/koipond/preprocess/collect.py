import pandas
import os
import lightkurve as lk
from lightkurve import LightCurveCollection
from koipond.preprocess.input import InputLightCurve, get_classification_label
from astropy.table import Table
import shutil
import time


def collect_data(raw_koi_file="", save_to=""):
    if os.path.exists(os.path.join(os.getcwd(), f"{save_to}/init/raw.csv")):
        downloaded_raw = pandas.read_csv(f"{save_to}/init/raw.csv")
    else:
        downloaded_raw = pandas.DataFrame({'curve_id':[], 'label':[]})
    already_done = set([int(cid) for cid in downloaded_raw['curve_id']])
    count = len([f for f in os.listdir(save_to) if f.endswith('.pt')])
    koi_df = pandas.read_csv(raw_koi_file)
    print(f"Reading {len(koi_df)} KOI from csv")
    for index, row in koi_df.iterrows():
        if row['kepid'] in already_done:
            print(f"Already completed {index}; skipping")
            continue
        current_lightcurves = get_lightcurves(kepid=row['kepid'])
        if current_lightcurves is None:
            continue
        print("Downloaded curve.")
        res_count = 0
        for res in current_lightcurves:
            res_count += 1
            res.to_pandas().to_csv(f"data/raw/{row['kepid']}_{res_count}.csv")
            # ic = InputLightCurve(kid=f"{row['kepid']}_{res_count}", lightcurve=res, label=get_classification_label(row['koi_disposition']))
            # ic.prepare()
            # ic.save_data(save_to)
        count+=1
        downloaded_raw = pandas.read_csv(filepath_or_buffer=f"data/init/raw.csv")
        downloaded_raw = pandas.concat((downloaded_raw, pandas.DataFrame({'curve_id':[row['kepid']],'label':[get_classification_label(row['koi_disposition'])]})), ignore_index=True)
        downloaded_raw.to_csv(path_or_buf=f"data/init/raw.csv", index=False)
        print(f"Saved total {count} kepler objects. ({index}/{len(koi_df)})")

        shutil.rmtree(lk.config.get_cache_dir(), ignore_errors=True)
        print("Cleared lightkurve cache.")
    print("DONE")

def get_lightcurves(kepid="", trytime=0):
    print("Querying KIC", kepid)
    try:
        search_result = lk.search_lightcurve(f"KIC{kepid}", author=('K2', 'Kepler'))
        downloaded = search_result.download_all()
        return downloaded
    except Exception as e:
        print(f"Failed to download lightkurve for", kepid)
        print(e)
        if trytime < 5:
            print(f"Trying again in 5 seconds. (attempt {trytime+1})")
            shutil.rmtree(lk.config.get_cache_dir(), ignore_errors=True)
            time.sleep(5)
            return get_lightcurves(kepid,trytime=trytime+1)
        else:
            print(f"Cannot retrieve lightcurve for KIC{kepid}. Skipping.")
            return None
        
def process_data(raw_csv="data/init/raw.csv", raw_dir="data/raw", labels_file="data/init/labels.csv", data_dir="data"):
    if os.path.exists(os.path.join(os.getcwd(), data_dir, "init/labels.csv")):
        current_labels_df = pandas.read_csv(f"{data_dir}/init/labels.csv")
        already_done = set([int(cid.replace(".pt", "")) for cid in current_labels_df['curve_id']])
    else:
        already_done = set()
    count = len([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    koi_df = pandas.read_csv(raw_csv)
    print(f"Reading {len(koi_df)} KOI from csv")
    # for index, row in koi_df.iterrows():
    completed_count = 0
    total_to_process = len([f for f in os.listdir(raw_dir) if f.endswith('.csv')])    
    for raw_file in os.listdir(raw_dir):
        if not raw_file.endswith('.csv'):
            continue
        curve_id = raw_file.split("_")[0]
        if curve_id in already_done:
            print(f"Already completed {raw_file.replace('.pt', '')}; skipping")
            continue
        label = int(koi_df.loc[koi_df['curve_id'] == int(curve_id)]['label'].tolist()[0])
        count += __prepare(file=f"{raw_dir}/{raw_file}", curve_id=raw_file.replace(".csv", ""), label=label, data_dir=data_dir)
        completed_count += 1
        print(f"Saved total {count} binned lightcurves. ({completed_count}/{total_to_process})")

def stack_and_process(raw_csv="data/init/raw.csv", raw_dir="data/raw", labels_file="data/init/labels.csv", data_dir="data"):
    if os.path.exists(os.path.join(os.getcwd(), data_dir, "init/labels.csv")):
        current_labels_df = pandas.read_csv(f"{data_dir}/init/labels.csv")
        already_done = set([int(cid.split("_")[0]) for cid in current_labels_df['curve_id']])
    else:
        already_done = set()
    count = len([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    raw_df = pandas.read_csv(raw_csv)
    print(f"Reading {len(raw_df)} KOI from csv")
    
    indexed_ids = {}
    all_raw_files = [f.split("_") for f in os.listdir(raw_dir) if f.endswith('.csv')]
    for curve_id in raw_df['curve_id'].tolist():
        key = int(curve_id)
        vals = ['_'.join(f) for f in all_raw_files if int(f[0]) == key]
        indexed_ids.update({key:vals})

    completed_count = len(already_done)
    total_to_process = len(raw_df)    
    for index, row in raw_df.iterrows():
        curve_id = row['curve_id']
        if  curve_id in already_done:
            print(f"Already completed {curve_id}; skipping")
            continue
        label = int(row['label'])

        curve_files = indexed_ids.get(curve_id, [])
        if len(curve_files) == 0:
            print("WARNING: Could not find curve files for", curve_id)
        lc_col = LightCurveCollection([])
        for c_file in curve_files:
            c_df = pandas.read_csv(f"{raw_dir}/{c_file}")
            lc = lk.LightCurve(data=Table.from_pandas(c_df))
            lc_col.append(lc)

        count += __prepare(lc_col=lc_col, curve_id=curve_id, label=label, data_dir=data_dir)
        completed_count += 1
        print(f"Saved total {count} binned lightcurves. ({completed_count}/{total_to_process})")


def __prepare(curve_id:str, label:int, file:str=None, lc_col:LightCurveCollection=None, data_dir="")->int:
    if file is not None:
        try:
            curve_df = pandas.read_csv(file)
            lightcurve = lk.LightCurve(data=Table.from_pandas(curve_df))
        except FileNotFoundError as e:
            return 0
    elif lc_col is not None:
        lightcurve = lc_col.stitch()

    ic = InputLightCurve(kid=f"{curve_id}", lightcurve=lightcurve, label=label)
    ic.prepare()
    print(f"Saving {len(ic.prepared_lightcurves)} files with ID {curve_id}") #debugging
    ic.save_data(data_dir)
   
    added = len(ic.prepared_lightcurves)

    # cleanup (does this help? who knows)
    if file is not None:
        del curve_df
    del lightcurve
    del ic

    return added

