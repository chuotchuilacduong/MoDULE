#!/usr/bin/env python3
"""Đặt lại tên W&B cho các run của main table theo quy ước
    {baseline}__{dataset}__{scenario}__{config_stem}__seed{seed}

Cần vì run_moe_pipeline.py ghi đè `study_name` bằng trường `name` cấp cao nhất,
nên các run pipeline log lên với tên kiểu `pacs_grip_class_baseline`. Metadata
trong wandb.config vẫn đúng nên tên chuẩn suy ra được -- không phải chạy lại run.

Hai quy tắc:
  * bỏ khoảng trắng trong tên baseline ("Gradient Ascent" -> "GradientAscent")
  * run không có metric nào (crash trước khi log) được thêm hậu tố __failed để
    không trùng tên với run thật cùng cấu hình
"""
import wandb

PROJ = "chuotchuilacduong-hanoi-university-of-science-and-technology/MoE"
api = wandb.Api()
changed = ok = 0
for r in api.runs(PROJ):
    c = r.config
    b, sc, cn = c.get("baseline_name"), c.get("scenario"), c.get("config_name")
    if not (b and sc and cn):
        continue
    want = f"{b.replace(' ', '')}__PACS__{sc}__{cn}__seed{c.get('seed', 42)}"
    if r.summary.get("fa") is None:          # crash trước khi có metric
        want += "__failed"
    if r.name == want:
        ok += 1
        continue
    print(f"  {r.name[:58]:<58} -> {want}   ({r.id}, {r.state})")
    r.name = want
    r.update()
    changed += 1
print(f"đổi tên: {changed} | đã đúng: {ok}")
