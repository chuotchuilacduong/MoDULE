"""Kéo bảng loss-weight sensitivity từ W&B (group LossWeight_PACS) và in LaTeX."""
import wandb
P="chuotchuilacduong-hanoi-university-of-science-and-technology/MoE"; api=wandb.Api()
combos=[(0,1,1),(1,0,1),(1,1,0),(0.1,1,1),(1,0.1,1),(1,1,0.1),(1,1,1)]
tag=lambda v: str(v).replace('.','p')
def latest(name):
    rs=[r for r in api.runs(P, filters={"display_name":name}, order="-created_at") if r.state=="finished"]
    return dict(rs[0].summary) if rs else {}
print(f"{'sp':>4s} {'bal':>4s} {'div':>4s} | {'CleanTA':>8s} {'RA':>7s} {'FA':>7s} {'TA':>7s} {'MIA':>6s}")
rows=[]
for sp,bal,div in combos:
    t=f"sp{tag(sp)}_bal{tag(bal)}_div{tag(div)}"
    b=latest(f"lossw_base_{t}"); u=latest(f"lossw_unl_class_{t}")
    cta=b.get("test_accuracy"); ra=u.get("ra"); fa=u.get("fa"); ta=u.get("ta"); mia=u.get("mia")
    f=lambda v,s=100: f"{v*s:.2f}" if isinstance(v,(int,float)) else "--"
    print(f"{sp:>4} {bal:>4} {div:>4} | {f(cta):>8s} {f(ra):>7s} {f(fa):>7s} {f(ta):>7s} {f(mia,1):>6s}")
    rows.append(f"{sp} & {bal} & {div} & {f(cta)} & {f(ra)} & {f(fa)} \\\\")
print("\n% LaTeX rows (Clean TA | RA | FA)"); print("\n".join(rows))
