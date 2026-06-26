import os, json
from urllib.parse import unquote
import requests
OUT = "data/diag/barchart_probe.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
FIELDS = "tradeTime.format(m/d/Y),openPrice,highPrice,lowPrice,lastPrice,priceChange,percentChange,volume,symbolCode,symbolType"
UAS = {
  "old_chrome":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
  "minimal":"Mozilla/5.0",
  "modern_chrome":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
}
def probe(sym, ua_key):
    ua = UAS[ua_key]; s = requests.Session()
    g = s.get(f"https://www.barchart.com/stocks/quotes/{sym}/price-history/",
              headers={"user-agent":ua,"referer":f"https://www.barchart.com/stocks/quotes/{sym}/price-history/historical"}, timeout=20)
    xsrf = unquote(unquote(s.cookies.get_dict().get("XSRF-TOKEN","")))
    r = s.get("https://www.barchart.com/proxies/core-api/v1/historical/get",
              params={"symbol":sym,"fields":FIELDS,"type":"eod","orderBy":"tradeTime","orderDir":"desc","limit":6,"raw":"1"},
              headers={"accept":"application/json","user-agent":ua,"x-xsrf-token":xsrf}, timeout=20)
    o = {"symbol":sym,"ua":ua_key,"get_status":g.status_code,"cookies":sorted(s.cookies.get_dict().keys()),
         "api_status":r.status_code,"api_len":len(r.content)}
    try: o["rows"] = r.json().get("data", [])[:6]
    except Exception as e: o["parse_error"]=str(e); o["body"]=r.text[:300]
    return o
res=[]
for sym in ["$SPX","$NSHU"]:
    for k in UAS:
        try: res.append(probe(sym,k))
        except Exception as e: res.append({"symbol":sym,"ua":k,"error":str(e)})
json.dump(res, open(OUT,"w"), indent=2)
print(json.dumps([{k:v for k,v in r.items() if k!="rows"} for r in res], indent=2))
