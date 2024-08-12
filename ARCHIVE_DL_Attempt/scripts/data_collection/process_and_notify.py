import koipond.preprocess.collect as collect
import http.client as http
import urllib.parse
import traceback, sys, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KOI Data.")
    parser.add_argument('-s','--stack', help='Stack all before processing', action='store_true')
    args = parser.parse_args()

    try:
        if args.stack:
            collect.stack_and_process(raw_csv="data/init/raw.csv", raw_dir="data/raw", labels_file="data/init/labels.csv", data_dir="data")
        else:
            collect.process_data(raw_csv="data/init/raw.csv", raw_dir="data/raw", labels_file="data/init/labels.csv", data_dir="data")
    except Exception as e:
        print(e)
        traceback.print_exception(*sys.exc_info()) 
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
                    "message":f"Error on processing data: {str(e)}",
                }), {"Content-type":"application/x-www-form-urlencoded"})
        res=conn.getresponse()
        print(res.read())
    else:
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
                    "message": "Process training data job finished.",
                }), {"Content-type":"application/x-www-form-urlencoded"})
        res=conn.getresponse()
        print(res.read())

