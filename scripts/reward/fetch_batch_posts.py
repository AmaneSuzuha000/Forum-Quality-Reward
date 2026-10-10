# -*- coding: utf-8 -*-
"""抓取近7天批次帖子的全部楼层, 用于计算 OP 创作量 (self_long_uniq)。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, setup_stdout

setup_stdout()

import asyncio

import aiotieba
import pandas as pd

D=ROOT/"data"/"reward"
b=pd.read_parquet(D/"recent_scored.parquet")
AUTH={int(r.tid):int(r.author_id) for r in b.itertuples()}
async def go():
    rows=[]
    async with aiotieba.Client() as c:
        for i,(tid,_au) in enumerate(AUTH.items()):
            pn=1
            while True:
                try:
                    r=await c.get_posts(tid,pn=pn)
                except Exception as e:
                    print("ERR",tid,pn,type(e).__name__,str(e)[:70],flush=True); break
                objs=getattr(r,"objs",None) or []
                if not objs: break
                for p in objs:
                    rows.append({"tid": tid,"pid": int(getattr(p,"pid",0) or 0),
                                     "author_id": int(getattr(p,"author_id",0) or 0),
                                     "text": str(getattr(p,"text","") or ""),
                                     "create_time": getattr(p,"create_time",None)})
                if len(objs)<30: break
                pn+=1
                if pn>15: break
                await asyncio.sleep(0.35)
            if (i+1)%25==0: print("  [%d/%d]"%(i+1,len(AUTH)),flush=True)
            await asyncio.sleep(0.45)
    return rows
rows=asyncio.run(go())
df=pd.DataFrame(rows)
df.to_parquet(D/"batch_posts.parquet",index=False)
print("posts",len(df),"threads",df["tid"].nunique(),"of",len(AUTH))
cov=df.groupby("tid").size()
print("per-thread floors: mean %.1f median %.0f max %d"%(cov.mean(),cov.median(),cov.max()))
