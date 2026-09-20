"""Bảng lr x gamma của l1-sparse (group L1Sweep_PACS_class) + chọn theo quy tắc: FA nhỏ nhất với RA >= 98, TA >= Original-2."""
import wandb
P="chuotchuilacduong-hanoi-university-of-science-and-technology/MoE"; api=wandb.Api()
rows=[]
for r in api.runs(P, filters={"group":"L1Sweep_PACS_class"}, order="+created_at"):
    s=dict(r.summary); c=r.config
    if r.state!="finished" or s.get("fa") is None: continue
    rows.append((c["lr"],c["alpha"],s["fa"]*100,s["ra"]*100,s["ta"]*100,s["mia"],s.get("epoch")))
print(f"{'lr':>8s} {'gamma':>8s} | {'FA':>6s} {'RA':>6s} {'TA':>6s} {'MIA':>6s} {'ep':>3s}")
for lr,g,fa,ra,ta,mia,ep in sorted(rows): print(f"{lr:8g} {g:8g} | {fa:6.2f} {ra:6.2f} {ta:6.2f} {mia:6.3f} {ep:>3}")
ok=[x for x in rows if x[3]>=98.0 and x[4]>=91.5]
if ok:
    b=min(ok,key=lambda x:x[2]); print(f"\nchọn (FA min, RA>=98, TA>=91.5): lr={b[0]:g} gamma={b[1]:g} -> FA {b[2]:.2f} RA {b[3]:.2f} TA {b[4]:.2f} MIA {b[5]:.3f}")
