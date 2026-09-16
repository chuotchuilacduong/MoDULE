"""So sánh hai run sep_off / sep_on: kéo online_spec/overall/* và loss cuối từ W&B, in bảng."""
import wandb
P = "chuotchuilacduong-hanoi-university-of-science-and-technology/MoE"
api = wandb.Api()
keys = ["mean_domain_mi_norm", "mean_domain_overlap", "mean_domain_topk_jaccard",
        "mean_class_mi_norm", "mean_class_overlap", "mean_class_topk_jaccard",
        "mean_domain_purity", "mean_domain_purity_used", "mean_class_purity",
        "mean_gate_entropy_norm", "total_dead_experts"]
rows = {}
for lam in ("0.0", "0.5"):
    runs = api.runs(P, filters={"display_name": f"septest_pacs_M8_k2_30ep_lsep{lam}"}, order="-created_at")
    runs = list(runs)
    if not runs:
        print(f"[!] không thấy run lsep{lam} trên W&B"); continue
    r = runs[0]; s = r.summary
    rows[lam] = {k: s.get(f"online_spec/overall/{k}") for k in keys}
    rows[lam].update({"sep_loss": s.get("sep_loss"), "ce_loss": s.get("ce_loss"),
                      "TA": s.get("ta"), "RA": s.get("ra"), "id": r.id, "state": r.state})
if len(rows) == 2:
    print(f"{'metric':32s} {'lambda_sep=0':>14s} {'lambda_sep=0.5':>14s} {'delta':>10s}")
    for k in keys + ["sep_loss", "ce_loss", "TA", "RA"]:
        a, b = rows["0.0"].get(k), rows["0.5"].get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            print(f"{k:32s} {a:14.4f} {b:14.4f} {b - a:+10.4f}")
        else:
            print(f"{k:32s} {str(a):>14s} {str(b):>14s}")
    print("\nKỳ vọng nếu L_sep hoạt động: mean_domain_mi_norm TĂNG, domain_overlap / topk_jaccard GIẢM,")
    print("total_dead_experts không tăng, TA/RA không giảm đáng kể. Nếu purity tăng mà dead_experts tăng và TA giảm")
    print("thì L_sep đang ép phân hoạch giả tạo.")
    for lam, d in rows.items(): print(f"  run lsep{lam}: https://wandb.ai/{P}/runs/{d['id']} ({d['state']})")
