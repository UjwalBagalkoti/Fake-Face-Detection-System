from __future__ import annotations
import argparse, csv
from pathlib import Path
IMAGE_EXTS={".jpg",".jpeg",".png",".webp"}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); root=Path(a.root); rows=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            parts=p.parts
            label="fake" if any(x.lower() in {"fake","manipulated","deepfake","forged"} for x in parts) else ("real" if any(x.lower() in {"real","original"} for x in parts) else "unknown")
            rows.append({"path":str(p),"label":label,"source_group":p.parent.name})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","label","source_group"]); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {a.output}")
if __name__=="__main__": main()
