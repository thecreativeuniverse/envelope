import pandas
import http.client as http
import urllib.parse
import traceback, sys, argparse, os
from koipond.estimation.lightcurve import KoiLightCurve
import lightkurve as lk 

def search(kepid, actual_period):
    try_count = 0
    lc_coll = None
    while lc_coll is None and try_count < 100:
        try:
            print("Searching kepid", kepid) 
            search_result = lk.search_lightcurve(f'KIC{kepid}', author=['Kepler', 'K2'], exptime='long')
            lc_coll = search_result.download_all()
        except Exception as e:
            print(e)
            try_count += 1
            print("Trying search again.")
    if lc_coll is None:
        print(f"Could not get search result for KIC{kepid}. Skipping.")
        return 'failed_search'
    if len(lc_coll) == 0:
        print("Could not find any long exposures for kepid", kepid)
        return
    koi_lc = KoiLightCurve(lc_collection=lc_coll, kepid=str(kepid))
    koi_lc.stitch(fn=lambda x: x.remove_nans().flatten().normalize(), sigma_lower=15, debug=args.debug)
    koi_lc.locate_possible_tps(debug=args.debug)
    folded_lc = koi_lc.find_time_periods(debug=args.debug)
    if folded_lc is None:
        print("No exoplanet transit found. Actual KOI period:", actual_period)
    else:
        print("Best fold period", folded_lc.estimated_tp, "Actual KOI period", actual_period)
        print(f"(regression score {folded_lc.regression_score})")
    print("========================================")
    return None if folded_lc is None else folded_lc.estimated_tp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Folding lets gooo.")
    parser.add_argument('-n','--notify', help='Notify on finish', action='store_true')
    parser.add_argument('-d','--debug', help='Notify on finish', action='store_true')
    parser.add_argument('-i','--kepid', help='Specific kepid')
    parser.add_argument('-o', '--output', help="Output results file", default="debug/output.txt")
    parser.add_argument('-c', '--csv', help="Results csv file", default="debug/results.csv")
    args = parser.parse_args()


    try:
        if not os.path.exists("debug"):
            os.mkdir("debug")
        if args.kepid is not None:
            df = pandas.read_csv('data/init/koi_labelled.csv')
            period = df.loc[df['kepid']==int(args.kepid)]['koi_period']
            search(str(args.kepid), period)
        else:
            output_file = open(os.path.join(os.getcwd(), args.output), 'a+') 
            filtered_koi_df = pandas.read_csv('data/init/confirmed_koi_w_params.csv')
            if not os.path.exists(os.path.join(os.getcwd(), args.csv)):
                results_csv = pandas.DataFrame({"kepid":[], "koi_period":[],
                                                "koi_disposition":[],"envolope_disposition":[],
                                                "envolope_period":[]})
                results_csv.to_csv(path_or_buf=args.csv)
            else:
                results_csv = pandas.read_csv(args.csv)
            count = 0
            for index, row in filtered_koi_df.iterrows(): 
                if count >= 9:
                    break
                kepid = row['kepid']
                if kepid in results_csv['kepid'].tolist():
                    print(f"Already searched kepid {kepid}; skipping")
                    continue
                count += 1
                actual_period = row['koi_period']
                is_confirmed = row['koi_disposition']
                try:
                    found_tp = search(kepid, actual_period)
                except Exception as e:
                    print(f"Could not find tp for KIC{kepid}")
                    found_tp = 'failed_search'
                output_file.write(f"Searching kepid {kepid}\n")
                output_file.write(f"koi_period = {actual_period}\n")
                output_file.write(f"disposition = {is_confirmed}\n")
                disposition = None
                if found_tp is None:
                    output_file.write(f"Found no lightcurve\n")
                    disposition = 'FALSE POSITIVE'
                elif found_tp == 'failed_search':
                    output_file.write("Could not retrieve light curve data; skipping.")
                else:
                    output_file.write(f"Found lightcurve of orbital period {found_tp}\n")
                    disposition = 'CONFIRMED'
                
                if disposition is not None:
                    df = pandas.DataFrame({"kepid":[kepid], "koi_period":[actual_period],
                                                "koi_disposition":[is_confirmed],"envolope_disposition":[disposition],
                                                "envolope_period":[found_tp]})
                    results_csv = pandas.concat((results_csv, df))
                    results_csv.to_csv(path_or_buf=args.csv)
                output_file.write("=-=-"*10)
                output_file.write("\n")
    except Exception as e:
        print(e)
        traceback.print_exception(*sys.exc_info()) 

        if not args.notify:
            exit()

        # this is mainly a temp solution. pushover tokens stored outside of git project.
        with open("../keys/api_token") as app_file:
            app_token = app_file.readline().replace("\n", "")
        with open("../keys/user_token") as user_file:
            user_token = user_file.readline().replace("\n", "")

        conn = http.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
                urllib.parse.urlencode({
                    "token": app_token,
                    "user": user_token,
                    "title": "Google Cloud",
                    "message":f"Error on processing transits: {str(e)}",
                }), {"Content-type":"application/x-www-form-urlencoded"})
        res=conn.getresponse()
    else:
        if args.notify:
            # this is mainly a temp solution. pushover tokens stored outside of git project.
            with open("../keys/api_token") as app_file:
                app_token = app_file.readline().replace("\n", "")
            with open("../keys/user_token") as user_file:
                user_token = user_file.readline().replace("\n", "")

            conn = http.HTTPSConnection("api.pushover.net:443")
            conn.request("POST", "/1/messages.json",
                    urllib.parse.urlencode({
                        "token": app_token,
                        "user": user_token,
                        "title": "Google Cloud",
                        "message": "Collect training data job finished.",
                    }), {"Content-type":"application/x-www-form-urlencoded"})
            res=conn.getresponse()
