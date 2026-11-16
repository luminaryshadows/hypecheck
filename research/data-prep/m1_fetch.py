import json, urllib.request, pandas as pd, os, time
UA={"User-Agent":"Mozilla/5.0"}
def get(u, timeout=60):
    req=urllib.request.Request(u, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
COLS=["ot","open","high","low","close","volume","ct","qv","n","tb","tq","ig"]
M=pd.read_csv("/home/ubuntu/Projects/hypecheck/data/memecoin-hype-dataset.csv")
M["avail_day"]=pd.to_datetime(M.avail_day)
pairs=sorted({(c,str(d.date())) for c,d in zip(M.coin, M.avail_day)})
print(f"{len(pairs)} (coin, day) pairs to fetch", flush=True)
done=0; fail=0
for coin, day in pairs:
    path=f"/tmp/m1/{coin}_{day}.csv"
    if os.path.exists(path): done+=1; continue
    import datetime as dt
    d0=dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
    s=int(d0.timestamp()*1000); e=s+86400_000
    try:
        rows=[]
        for a,b in [(s, s+1000*60000-1), (s+1000*60000, e)]:
            rows+=get(f"https://api.binance.com/api/v3/klines?symbol={coin}USDT&interval=1m&startTime={a}&endTime={b}&limit=1000")
        if not rows: fail+=1; continue
        df=pd.DataFrame(rows, columns=COLS)[["ot","open","high","low","close","volume"]]
        df.to_csv(path, index=False); done+=1
        if done % 100 == 0: print(f"  {done}/{len(pairs)} fetched ({fail} empty)", flush=True)
    except Exception as ex:
        fail+=1
        if fail < 6: print(f"  {coin} {day} FAILED {type(ex).__name__}", flush=True)
    time.sleep(0.05)
print(f"DONE {done}/{len(pairs)} files, {fail} failures", flush=True)
