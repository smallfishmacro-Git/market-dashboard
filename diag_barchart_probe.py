import os, json
from urllib.parse import unquote
import requests
os.makedirs("data/diag", exist_ok=True)
F = "tradeTime.format(m/d/Y),openPrice,highPrice,lowPrice,lastPrice,priceChange,percentChange,volume,symbolCode,symbolType"
UA_S = "Mozilla/5.0"
UA_OLD = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36"
GS = lambda s: {"user-agent":UA_S,"referer":f"https://www.barchart.com/stocks/quotes/{s}/price-history/historical"}
GF = lambda s: {"accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
  "accept-encoding":"gzip, deflate, br","accept-language":"en-US,en;q=0.9","cache-control":"max-age=0",
  "upgrade-insecure-requests":"1","referer":f"https://www.barchart.com/stocks/quotes/{s}/price-history/historical",
  "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}
def fetch(sym, limit, geth, ua, sess=None):
    s = sess or requests.Session()
    s.get(f"https://www.barchart.com/stocks/quotes/{sym}/price-history/", headers=geth(sym), timeout=20)
    ah = {"accept":"application/json","user-agent":ua,"x-xsrf-token":unquote(unquote(s.cookies.get_dict().get("XSRF-TOKEN","")))}
    r = s.get("https://www.barchart.com/proxies/core-api/v1/historical/get",
        params={"symbol":sym,"fields":F,"type":"eod","orderBy":"tradeTime","orderDir":"desc","limit":limit,"raw":"1"},
        headers=ah, timeout=20)
    try: rows = r.json().get("data", [])
    except: rows = []
    g = lambda x,k,i: (x.get(k) if isinstance(x,dict) else (x[i] if len(x)>i else None))
    return {"status":r.status_code,"n":len(rows),"last":[g(x,"lastPrice",4) for x in rows[:3]],"high":[g(x,"highPrice",2) for x in rows[:3]]}
res = {}
res["A_simple_lim6_baseline"] = fetch("$NSHU", 6, GS, UA_S)
res["B_simple_lim65"]         = fetch("$NSHU", 65, GS, UA_S)
res["C_simple_lim120"]        = fetch("$NSHU", 120, GS, UA_S)
res["D_fullheaders_lim6"]     = fetch("$NSHU", 6, GF, UA_OLD)
res["E_pipeline_full_lim120"] = fetch("$NSHU", 120, GF, UA_OLD)
s = requests.Session()
for sym in ["$SPX","$VIX","$VXV","$MMTH","$MMFI","$MMOH","$CPCS","$CPC","$MAHP","$MALP","$NAHC","$NALC","$M1HN","$M1LN","$UNCN"]:
    fetch(sym, 120, GS, UA_S, s)
res["F_after_burst15_lim6"]   = fetch("$NSHU", 6, GS, UA_S, s)
json.dump(res, open("data/diag/barchart_probe.json","w"), indent=2)
print(json.dumps(res, indent=2))
