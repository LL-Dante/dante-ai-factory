#!/usr/bin/env python3
"""Local-only, read-only hardware telemetry service. Python standard library only."""
import csv, ctypes, json, os, re, shutil, signal, subprocess, sys, threading, time
from collections import deque
from datetime import datetime, timezone, timedelta
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(r"C:\DanteAI\monitoring")
CONFIG = ROOT / "config" / "thresholds.json"
STATE = ROOT / "state" / "current.json"
TELEMETRY = ROOT / "logs" / "telemetry"
ALERTS = ROOT / "logs" / "alerts"
WEB = ROOT / "dashboard"
PHYSICAL_DISKS_PS = "Get-PhysicalDisk -ErrorAction SilentlyContinue | Select-Object Model,FriendlyName,HealthStatus,OperatingStatus,SerialNumber,Size,BusType,MediaType | ConvertTo-Json -Compress -Depth 3"
FALLBACK_DISKS = [{"Model":"Lexar SSD NM790 2TB","FriendlyName":"Lexar SSD NM790 2TB","HealthStatus":"Healthy"}]
PY = shutil.which("python") or sys.executable
lock, stop = threading.Lock(), threading.Event()
latest = {"timestamp": None, "overall_status": "NORMAL", "sensors": {}, "history": {}}
history = {k: deque(maxlen=900) for k in ("cpu.temperature_c", "gpu.temperature_c", "storage.lexar_nm790.temperature_c", "gpu.power_w", "gpu.utilization_percent", "gpu.vram_used_mb", "memory.used_mb")}
last_alert, active = {}, {}

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def ps(command):
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception: return None
def number(v):
    try: return float(str(v).strip().replace(",", "."))
    except Exception: return None
def read_config():
    try: return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception: return {"poll_seconds":2,"telemetry_log_seconds":60,"retention_days":30,"alert_cooldown_seconds":900,"thresholds":{}}
def gpu():
    names="temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,fan.speed,memory.used,memory.total"
    try:
        p=subprocess.run(["nvidia-smi",f"--query-gpu={names}","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=8,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        vals=[number(x) for x in p.stdout.strip().split(",")]
        if p.returncode or len(vals)!=8: raise RuntimeError(p.stderr)
        keys=("temperature_c","power_w","power_limit_w","utilization_percent","memory_utilization_percent","fan_percent","vram_used_mb","vram_total_mb")
        d=dict(zip(keys,vals)); d["vram_utilization_percent"]=round(100*d["vram_used_mb"]/d["vram_total_mb"],1) if d["vram_total_mb"] else None
        d["source"]="nvidia-smi"; return d
    except Exception as e: return {"source":"nvidia-smi","available":False,"error":str(e)[:160]}
def cstr(raw): return bytes(raw).split(b"\0",1)[0].decode("utf-8","replace")
class Hdr(ctypes.LittleEndianStructure):
    _pack_=1; _fields_=[("sig",ctypes.c_uint32),("version",ctypes.c_uint32),("revision",ctypes.c_uint32),("updated",ctypes.c_int64),("sensor_offset",ctypes.c_uint32),("sensor_size",ctypes.c_uint32),("sensor_count",ctypes.c_uint32),("reading_offset",ctypes.c_uint32),("reading_size",ctypes.c_uint32),("reading_count",ctypes.c_uint32)]
class Sensor(ctypes.LittleEndianStructure):
    _pack_=1; _fields_=[("id",ctypes.c_uint32),("instance",ctypes.c_uint32),("original",ctypes.c_char*128),("user",ctypes.c_char*128)]
class Reading(ctypes.LittleEndianStructure):
    _pack_=1; _fields_=[("kind",ctypes.c_uint32),("sensor_index",ctypes.c_uint32),("id",ctypes.c_uint32),("original",ctypes.c_char*128),("user",ctypes.c_char*128),("unit",ctypes.c_char*16),("value",ctypes.c_double),("minimum",ctypes.c_double),("maximum",ctypes.c_double),("average",ctypes.c_double)]
def hwinfo_cpu_temperature():
    """Read the official HWiNFO SM2 interface; unavailable is represented as None."""
    k=ctypes.WinDLL("kernel32",use_last_error=True); h=k.OpenFileMappingW(4,False,r"Global\HWiNFO_SENS_SM2")
    if not h: return None,"HWiNFO shared memory unavailable"
    k.MapViewOfFile.restype=ctypes.c_void_p; p=k.MapViewOfFile(h,4,0,0,0)
    if not p: k.CloseHandle(h); return None,"HWiNFO mapping could not be read"
    try:
        hdr=Hdr.from_address(p)
        if hdr.sig != 0x53695748: return None,"HWiNFO shared memory inactive"
        owners=[(cstr((s:=Sensor.from_address(p+hdr.sensor_offset+i*hdr.sensor_size)).user) or cstr(s.original)) for i in range(min(hdr.sensor_count,4096))]
        for i in range(min(hdr.reading_count,16384)):
            r=Reading.from_address(p+hdr.reading_offset+i*hdr.reading_size); label=cstr(r.user) or cstr(r.original); owner=owners[r.sensor_index] if r.sensor_index<len(owners) else ""
            if r.kind==1 and "CPU" in owner.upper() and label.strip().casefold() in ("cpu (tctl/tdie)","cpu package") and 0<r.value<130: return round(r.value,1),f"HWiNFO shared memory: {owner} / {label}"
        return None,"CPU (Tctl/Tdie) not exposed by HWiNFO"
    except Exception as e: return None,f"HWiNFO read error: {str(e)[:100]}"
    finally: k.UnmapViewOfFile(ctypes.c_void_p(p)); k.CloseHandle(h)
def system():
    mem=ps("$x=Get-CimInstance Win32_OperatingSystem; [pscustomobject]@{total=[math]::Round($x.TotalVisibleMemorySize/1024);available=[math]::Round($x.FreePhysicalMemory/1024)} | ConvertTo-Json -Compress")
    cpu=ps("(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime")
    try: m=json.loads(mem); total, avail=int(m["total"]), int(m["available"])
    except Exception: total=avail=None
    temp,source=hwinfo_cpu_temperature()
    return {"cpu":{"utilization_percent":number(cpu),"temperature_c":temp,"package_power_w":None,"source":source+"; utilization: Windows performance counters"},"memory":{"total_mb":total,"available_mb":avail,"used_mb":(total-avail if total is not None else None),"utilization_percent":round(100*(total-avail)/total,1) if total else None,"source":"Win32_OperatingSystem"}}
def storage_raw():
    return ps(PHYSICAL_DISKS_PS)
def slugify(label):
    s=re.sub(r"[^a-z0-9]+","_",label.casefold()).strip("_")
    return (s or "disk")[:24]
def storage_key(name):
    """NM790 keeps its canonical key (config/dashboard/thresholds compat); other disks are slugified."""
    return "lexar_nm790" if "NM790" in name.upper() else slugify(name)
def parse_storage(raw):
    """Pure parser for Get-PhysicalDisk JSON: returns {key:{model,health,size_gb,bus_type,media_type,temperature_c,secondary_temperature_c,source}} or None when unusable."""
    if raw is None: return None
    try: disks=json.loads(raw)
    except Exception: return None
    if isinstance(disks,dict): disks=[disks]
    if not isinstance(disks,list): return None
    out,used={},set()
    for d in disks:
        if not isinstance(d,dict): continue
        name=(d.get("Model") or d.get("FriendlyName") or "").strip()
        if not name: continue
        op=str(d.get("OperatingStatus") or "")
        if op!="": health="Degraded" if op.lower() in ("degraded","pred_fail") else ("Healthy" if op=="OK" else op.title())
        else: health="Unknown" if not str(d.get("HealthStatus") or "") else str(d.get("HealthStatus")).title()
        key=storage_key(name)
        if key in used:
            k=2
            while f"{key}_{k}" in used: k+=1
            key=f"{key}_{k}"
        used.add(key)
        size=d.get("Size")
        out[key]={"model":name,"health":health,"size_gb":round(size/1024**3,1) if isinstance(size,(int,float)) else None,"bus_type":d.get("BusType") or None,"media_type":d.get("MediaType") or None,"temperature_c":None,"secondary_temperature_c":None,"source":"Get-PhysicalDisk; temperature unavailable (StorageReliabilityCounter not available)"}
    return out if out else None
def storage_fallback():
    return parse_storage(json.dumps(FALLBACK_DISKS))
def storage():
    d=parse_storage(storage_raw())
    if d is None: d=storage_fallback()
    return d
def setpath(root,path,val):
    for part in path.split(".")[:-1]: root=root.setdefault(part,{})
    root[path.split(".")[-1]]=val
def getpath(root,path):
    for part in path.split("."):
        if not isinstance(root,dict): return None
        root=root.get(part)
    return root
def log_alert(row):
    f=ALERTS/f"alerts-{datetime.now():%Y-%m-%d}.csv"; exists=f.exists()
    with f.open("a",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["timestamp","sensor","value","threshold","severity","message"]); 
        if not exists:w.writeheader()
        w.writerow(row)
def evaluate(data,cfg):
    overall="NORMAL"; order={"NORMAL":0,"WARNING":1,"CRITICAL":2}; t=cfg["thresholds"]; cooldown=cfg["alert_cooldown_seconds"]; stamp=time.time()
    for path,limits in t.items():
        value=getpath(data,path)
        if value is None: continue
        sev="CRITICAL" if value>=limits["critical"] else "WARNING" if value>=limits["warning"] else "NORMAL"
        if order[sev]>order[overall]: overall=sev
        prior=active.get(path,"NORMAL")
        if sev != "NORMAL" and (sev!=prior or stamp-last_alert.get(path,0)>=cooldown):
            log_alert({"timestamp":now(),"sensor":path,"value":value,"threshold":limits[sev.lower()],"severity":sev,"message":f"{path} is {sev.lower()}"}); last_alert[path]=stamp
        if sev=="NORMAL" and prior!="NORMAL": log_alert({"timestamp":now(),"sensor":path,"value":value,"threshold":limits["warning"],"severity":"RECOVERY","message":f"{path} returned to normal"})
        active[path]=sev
    return overall
def compact(data):
    return {"timestamp":data["timestamp"],"cpu_temperature_c":getpath(data,"cpu.temperature_c"),"gpu_temperature_c":getpath(data,"gpu.temperature_c"),"nvme_temperature_c":getpath(data,"storage.lexar_nm790.temperature_c"),"gpu_power_w":getpath(data,"gpu.power_w"),"gpu_utilization_percent":getpath(data,"gpu.utilization_percent"),"vram_used_mb":getpath(data,"gpu.vram_used_mb"),"ram_used_mb":getpath(data,"memory.used_mb")}
def telemetry_log(data):
    f=TELEMETRY/f"telemetry-{datetime.now():%Y-%m-%d}.csv"; exists=f.exists(); row=compact(data)
    with f.open("a",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=row.keys());
        if not exists:w.writeheader()
        w.writerow(row)
def cleanup(days):
    cutoff=time.time()-days*86400
    for folder in (TELEMETRY,ALERTS):
        for f in folder.glob("*.csv"):
            try:
                if f.stat().st_mtime<cutoff:f.unlink()
            except OSError: pass
def poller():
    nextlog=0; lastclean=0
    while not stop.is_set():
        cfg=read_config(); s=system(); data={"timestamp":now(),"cpu":s["cpu"],"gpu":gpu(),"memory":s["memory"],"storage":storage(),"motherboard":{"source":"HWiNFO shared memory unavailable; no values exposed"}}
        data["overall_status"]=evaluate(data,cfg)
        point_time=int(time.time()*1000)
        for key,q in history.items():
            v=getpath(data,key); q.append([point_time,v])
        data["history"]={k:list(v) for k,v in history.items()}; data["sensor_sources"]={"gpu":"nvidia-smi","cpu_utilization":"Windows PerfFormattedData","memory":"Win32_OperatingSystem","storage_identity_health":"Get-PhysicalDisk","cpu_nvme_motherboard_temperatures":"Unavailable: HWiNFO shared memory disabled; Windows source unavailable"}
        if time.time()>=nextlog: telemetry_log(data); nextlog=time.time()+cfg["telemetry_log_seconds"]
        if time.time()-lastclean>3600: cleanup(cfg["retention_days"]); lastclean=time.time()
        with lock:
            global latest; latest=data
            tmp=STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(data,indent=2),encoding="utf-8"); os.replace(tmp,STATE)
        stop.wait(max(1,cfg["poll_seconds"]))
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/api/status","/api/status/"):
            with lock: body=json.dumps(latest).encode()
            self.send_response(200); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Cache-Control","no-store"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
        return super().do_GET()
    def log_message(self,*args): pass
def main():
    os.chdir(WEB); threading.Thread(target=poller,daemon=True).start(); server=ThreadingHTTPServer(("127.0.0.1",8765),Handler); server.daemon_threads=True
    def quit(*_): stop.set(); server.shutdown()
    signal.signal(signal.SIGINT,quit); signal.signal(signal.SIGTERM,quit)
    print("Dante monitoring: http://127.0.0.1:8765",flush=True); server.serve_forever()
if __name__=="__main__": main()
