import koipond.preprocess.collect as collect
import http.client as http
import urllib.parse

if __name__ == "__main__":
    try:
        collect.collect_data(raw_koi_file="data/init/koi_labelled.csv", save_to="data")
    except Exception as e:
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
                    "message":f"Error on collecting data: {str(e)}",
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
                    "message": "Collect training data job finished.",
                }), {"Content-type":"application/x-www-form-urlencoded"})
        res=conn.getresponse()
        print(res.read())

