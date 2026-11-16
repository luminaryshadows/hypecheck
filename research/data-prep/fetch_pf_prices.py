import json, urllib.request, urllib.parse, time, datetime as dt, csv, os
UA={"User-Agent":"Mozilla/5.0","Accept":"application/json"}
OUT="/tmp/pf-prices"
def get(u, timeout=90):
    req=urllib.request.Request(u, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
def resolve(sym):
    d=get("https://api.coingecko.com/api/v3/search?query="+urllib.parse.quote(sym))
    for c in d.get("coins",[]):
        if c["symbol"].upper()==sym.upper(): return c["id"]
    return d["coins"][0]["id"] if d.get("coins") else None
TARGETS={"FWOG":"fwog","POPCAT":"popcat","FARTCOIN":None,"GOAT":None,"MOG":None,
         "GIGA":None,"MOODENG":None,"CHILLGUY":None,"SLERF":None}
for sym, cid in TARGETS.items():
    path=os.path.join(OUT, f"{sym}-cg.csv")
    if os.path.exists(path):
        print(f"  {sym:9s} already fetched"); continue
    try:
        if cid is None:
            cid=resolve(sym); time.sleep(35)
            print(f"  {sym:9s} resolved id -> {cid}")
        d=get(f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart?vs_currency=usd&days=365&interval=daily")
        p=d.get("prices",[])
        if not p:
            print(f"  {sym:9s} no prices returned"); time.sleep(35); continue
        with open(path,"w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["date","close_usd"])
            for ms,price in p:
                w.writerow([str(dt.datetime.utcfromtimestamp(ms/1000).date()), price])
        print(f"  {sym:9s} {len(p)} daily closes -> {path}")
    except Exception as e:
        print(f"  {sym:9s} FAILED {type(e).__name__}: {str(e)[:70]}")
    time.sleep(35)
print("done")
