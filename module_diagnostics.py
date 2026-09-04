import os
import argparse
import random
from pathlib import Path
import yaml
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset

from dataset.transform.test_transform import get_test_transform
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform

from architecture.deity import DeiTArchitecture
from architecture.resnet import ResNetArchitecture
from architecture.module import ModuleArchitecture
from architecture.erm_ktp_resnet import ERM_KTP_Resnet
from architecture.asu_deity import ASUDeiTArchitecture 


def _wandb_settings():
    """Build wandb Settings that work across wandb versions.

    Older wandb accepted Settings(start_method='thread'); newer releases removed
    the field and their pydantic-validated Settings rejects it outright
    ("Extra inputs are not permitted"). Their default already runs the client in
    a thread, so falling back to None is equivalent.
    """
    try:
        return wandb.Settings(start_method='thread')
    except Exception:
        return None

class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        data = self.subset[idx]
        image = self.resize(data[0])
        if self.transform:
            image = self.transform(image)
        return (image,) + data[1:]

    def __len__(self):
        return len(self.subset)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# safely get domain from dataset.
def get_domain(dataset, idx):
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    data_tuple = dataset[idx]
    domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(domain_val)

@torch.no_grad()
# funtion for evaluate acc.
def evaluate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits, _ = model.inference(images)
        
        predictions = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    
    return 100 * correct / total

def extract_routing_info(model, dataloader, device, num_experts, gate_k):
    all_probs = []

    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    if len(moe_layers) == 0:
        raise ValueError("[!] No DeepMoELayer found in the model!")

    model.eval() 
    with torch.no_grad(): 
        for batch in dataloader:
            images = batch[0].to(device)
            B = images.size(0) 
            
            _ = model.inference(images) 
            
            batch_pi_layers = []
            
            for moe_layer in moe_layers:
                pi_flat = moe_layer.last_pi_all 
                S = pi_flat.size(0) // B 
                # (B*S, E) -> (B, S, E) -> (B, E)
                pi_layer = pi_flat.view(B, S, num_experts).mean(dim=1)
                batch_pi_layers.append(pi_layer)
                
            # collect all layers -> (L, B, num_experts).
            batch_pi_stacked = torch.stack(batch_pi_layers, dim=0)
            all_probs.append(batch_pi_stacked.cpu())
            
    all_probs_cat = torch.cat(all_probs, dim=1)
    _, all_topk_indices = torch.topk(all_probs_cat, k=gate_k, dim=-1)

    return all_probs_cat, all_topk_indices

def _entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(dim=-1)


@torch.no_grad()
def extract_router_diagnostics(model, dataloader, device, num_experts, gate_k, num_classes, num_domains):
    """
    Single pass over `dataloader` (batches of (images, labels, domains)) collecting,
    per MoE layer: token-level sparsity entropy (both last_pi_all and last_pi),
    all-token vs CLS-token routing mass bucketed by domain, and per-expert
    domain/class hit tallies from the per-image top-gate_k assignment (same
    per-image averaging convention as `extract_routing_info`).
    """
    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    if len(moe_layers) == 0:
        raise ValueError("[!] No DeepMoELayer found in the model!")
    L = len(moe_layers)

    token_entropy_sum = torch.zeros(L)
    token_entropy_count = torch.zeros(L)
    gatek_entropy_sum = torch.zeros(L)
    gatek_entropy_count = torch.zeros(L)
    mass_all_sum = torch.zeros(L, num_domains, num_experts)
    token_count_per_domain = torch.zeros(L, num_domains)
    mass_cls_sum = torch.zeros(L, num_domains, num_experts)
    image_count_per_domain = torch.zeros(L, num_domains)
    domain_tally = torch.zeros(L, num_experts, num_domains)
    class_tally = torch.zeros(L, num_experts, num_classes)

    model.eval()
    for batch in dataloader:
        images = batch[0].to(device)
        labels = batch[1].to(device).long()
        domains = batch[2].to(device).long()
        B = images.size(0)

        _ = model.inference(images)

        for l, m in enumerate(moe_layers):
            pi_flat = m.last_pi_all              # [B*S, E]
            gk_flat = m.last_pi                  # [B*S, gate_k]
            S = pi_flat.size(0) // B

            ent_all = _entropy(pi_flat)
            token_entropy_sum[l] += ent_all.sum().item()
            token_entropy_count[l] += ent_all.numel()

            ent_gk = _entropy(gk_flat)
            gatek_entropy_sum[l] += ent_gk.sum().item()
            gatek_entropy_count[l] += ent_gk.numel()

            domain_tok = domains.repeat_interleave(S).cpu()
            pi_flat_cpu = pi_flat.detach().cpu()
            mass_all_sum[l].index_add_(0, domain_tok, pi_flat_cpu)
            token_count_per_domain[l].index_add_(0, domain_tok, torch.ones(domain_tok.size(0)))

            pi_bse = pi_flat.view(B, S, num_experts)
            cls_pi = pi_bse[:, 0, :].detach().cpu()
            domains_cpu = domains.cpu()
            mass_cls_sum[l].index_add_(0, domains_cpu, cls_pi)
            image_count_per_domain[l].index_add_(0, domains_cpu, torch.ones(B))

            # per-image expert assignment: average pi over tokens, then top-gate_k
            # (matches extract_routing_info's convention, so tallies stay consistent
            # with the retain/forget hit-rate machinery reused later).
            pi_img = pi_bse.mean(dim=1)  # [B, E]
            _, topk_idx = pi_img.topk(gate_k, dim=-1)  # [B, gate_k]
            topk_idx_cpu = topk_idx.cpu()
            labels_cpu = labels.cpu()
            for k in range(gate_k):
                expert_idx = topk_idx_cpu[:, k]
                dflat = expert_idx * num_domains + domains_cpu
                domain_tally[l].view(-1).index_add_(0, dflat, torch.ones(B))
                cflat = expert_idx * num_classes + labels_cpu
                class_tally[l].view(-1).index_add_(0, cflat, torch.ones(B))

    return {
        "num_layers": L,
        "token_entropy_sum": token_entropy_sum, "token_entropy_count": token_entropy_count,
        "gatek_entropy_sum": gatek_entropy_sum, "gatek_entropy_count": gatek_entropy_count,
        "mass_all_sum": mass_all_sum, "token_count_per_domain": token_count_per_domain,
        "mass_cls_sum": mass_cls_sum, "image_count_per_domain": image_count_per_domain,
        "domain_tally": domain_tally, "class_tally": class_tally,
    }


def print_router_diagnostics(diag, num_experts, gate_k, domain_names, class_names,
                              forget_domain_idx=0, dead_expert_threshold=0.01, log_to_wandb=False,
                              split_label="TRAIN"):
    L = diag["num_layers"]
    num_domains = len(domain_names)
    log_E = float(np.log(num_experts))
    log_gk = float(np.log(gate_k)) if gate_k > 1 else 0.0
    wandb_prefix = f"router_{split_label.lower()}"

    for l in range(L):
        print(f"\n{'='*70}\nMoE layer {l}  [{split_label} split]\n{'='*70}")

        te = (diag["token_entropy_sum"][l] / diag["token_entropy_count"][l].clamp(min=1)).item()
        ge = (diag["gatek_entropy_sum"][l] / diag["gatek_entropy_count"][l].clamp(min=1)).item()
        print(f"[Sparsity] all-token entropy (last_pi_all): {te:.4f} / max log({num_experts})={log_E:.4f}"
              f"  -- descriptive only, Lsp does NOT train on this")
        print(f"[Sparsity] top-{gate_k} gate entropy (last_pi):    {ge:.4f} / max log({gate_k})={log_gk:.4f}"
              f"  -- this IS what Lsp trains on")

        mass_all_pd = diag["mass_all_sum"][l] / diag["token_count_per_domain"][l].clamp(min=1).unsqueeze(-1)
        mass_cls_pd = diag["mass_cls_sum"][l] / diag["image_count_per_domain"][l].clamp(min=1).unsqueeze(-1)
        mass_all_global = diag["mass_all_sum"][l].sum(0) / diag["token_count_per_domain"][l].sum().clamp(min=1)

        dead = (mass_all_global < dead_expert_threshold).sum().item()
        balance_dev = ((mass_all_global - 1.0 / num_experts) ** 2).sum().item()
        print(f"[Load] global all-token mass per expert: {[round(x, 4) for x in mass_all_global.tolist()]}")
        print(f"[Load] dead experts (<{dead_expert_threshold*100:.0f}% mass): {dead} / {num_experts}"
              f"   | balance deviation (untrained-for): {balance_dev:.6f}")

        if log_to_wandb:
            wandb.log({
                f"{wandb_prefix}/layer{l}/token_entropy": te,
                f"{wandb_prefix}/layer{l}/token_entropy_max": log_E,
                f"{wandb_prefix}/layer{l}/gatek_entropy": ge,
                f"{wandb_prefix}/layer{l}/gatek_entropy_max": log_gk,
                f"{wandb_prefix}/layer{l}/dead_experts": dead,
                f"{wandb_prefix}/layer{l}/balance_deviation": balance_dev,
            })

        print(f"\n[Per-domain all-token routing mass]  (rows sum to ~1)")
        header = "  domain".ljust(14) + "".join(f"e{e}".rjust(8) for e in range(num_experts))
        print(header)
        mass_rows = []
        for d in range(num_domains):
            row = "".join(f"{mass_all_pd[d, e].item():8.4f}" for e in range(num_experts))
            tag = "  <- forget" if d == forget_domain_idx else ""
            print(f"  {domain_names[d]:<12}{row}{tag}")
            mass_rows.append([domain_names[d]] + [mass_all_pd[d, e].item() for e in range(num_experts)])

        print(f"\n[CLS-token vs all-token mass, per domain]  (L1 distance = |CLS - all| summed over experts)")
        l1_rows = []
        for d in range(num_domains):
            l1 = (mass_cls_pd[d] - mass_all_pd[d]).abs().sum().item()
            print(f"  {domain_names[d]:<12} L1 distance: {l1:.4f}"
                  f"   (large => Eq.7's all-token score diverges from what the classifier actually sees)")
            l1_rows.append([domain_names[d], l1])

        print(f"\n[Per-expert domain/class specialization]  (from per-image top-{gate_k} assignment)")
        dom_tally = diag["domain_tally"][l]   # (E, D)
        cls_tally = diag["class_tally"][l]    # (E, C)
        header = f"  {'expert':<8}{'load%':>8}{'top domain':>16}{'dom purity':>12}{'top class':>16}{'cls purity':>12}"
        print(header)
        spec_rows = []
        for e in range(num_experts):
            dsum = dom_tally[e].sum().item()
            csum = cls_tally[e].sum().item()
            if dsum > 0:
                d_share = dom_tally[e] / dsum
                top_d = domain_names[int(d_share.argmax().item())]
                d_purity = d_share.max().item()
            else:
                top_d, d_purity = "-", 0.0
            if csum > 0:
                c_share = cls_tally[e] / csum
                top_c = class_names[int(c_share.argmax().item())]
                c_purity = c_share.max().item()
            else:
                top_c, c_purity = "-", 0.0
            load_pct = mass_all_global[e].item() * 100
            print(f"  {e:<8}{load_pct:8.2f}{top_d:>16}{d_purity:12.3f}{top_c:>16}{c_purity:12.3f}")
            spec_rows.append([e, load_pct, top_d, d_purity, top_c, c_purity])

        if log_to_wandb:
            wandb.log({
                f"{wandb_prefix}/layer{l}/domain_mass_table": wandb.Table(
                    columns=["domain"] + [f"e{e}" for e in range(num_experts)], data=mass_rows),
                f"{wandb_prefix}/layer{l}/cls_vs_all_l1_table": wandb.Table(
                    columns=["domain", "l1_distance"], data=l1_rows),
                f"{wandb_prefix}/layer{l}/expert_specialization_table": wandb.Table(
                    columns=["expert", "load_pct", "top_domain", "dom_purity", "top_class", "cls_purity"],
                    data=spec_rows),
            })


def compute_localization_diagnostics(model, forget_loader, retain_loader, device, num_experts, gate_k, k_u=2, alpha=1.0):
    print(f"[*] Extracting routing info... (Using alpha={alpha}, k_u={k_u})")
    
    pi_f, E_f = extract_routing_info(model, forget_loader, device, num_experts, gate_k)
    pi_r, E_r = extract_routing_info(model, retain_loader, device, num_experts, gate_k)

    L = pi_f.size(0) # number of MoE layers.
    N_f = pi_f.size(1)
    N_r = pi_r.size(1)

    total_FEM, total_ARR, total_RFO = 0.0, 0.0, 0.0

    # caculate metric for each layer.
    for l in range(L):
        pi_f_l, E_f_l = pi_f[l], E_f[l]
        pi_r_l, E_r_l = pi_r[l], E_r[l]

        mass_f = pi_f_l.mean(dim=0)
        mass_r = pi_r_l.mean(dim=0)

        # scoring k_u by difference.
        diff = mass_f - (alpha * mass_r)
        _, M_f = torch.topk(diff, k=k_u, dim=-1)

        # forget-expert routing mass (FEM).
        FEM_l = pi_f_l[:, M_f].sum(dim=1).mean().item()
        total_FEM += FEM_l

        # at-risk retain ratio (ARR).
        overlap_mask = torch.isin(E_r_l, M_f).any(dim=1)
        ARR_l = overlap_mask.float().mean().item()
        total_ARR += ARR_l

        # retain-forget routing overlap (RFO).
        E_f_hot = torch.zeros((N_f, num_experts), dtype=torch.float)
        E_r_hot = torch.zeros((N_r, num_experts), dtype=torch.float)
        
        E_f_hot.scatter_(1, E_f_l, 1.0)
        E_r_hot.scatter_(1, E_r_l, 1.0)

        intersection = E_f_hot @ E_r_hot.T 
        sum_f = E_f_hot.sum(dim=1, keepdim=True) 
        sum_r = E_r_hot.sum(dim=1, keepdim=True) 
        union = sum_f + sum_r.T - intersection
        
        iou_matrix = intersection / torch.clamp(union, min=1e-8)
        RFO_l = iou_matrix.mean().item()
        total_RFO += RFO_l

    # average all dataset.
    return total_FEM / L, total_ARR / L, total_RFO / L

def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument('--config', type=str, default=None,
                         help="Path to a flat config .yaml. If omitted, derived from "
                              "<checkpoint>/../../learn.yaml (the run_moe_pipeline.py convention).")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the .pth checkpoint file.")
    parser.add_argument('--run-eq7-diagnostics', action='store_true',
                         help="Also run the existing FEM/ARR/RFO localization diagnostics "
                              "(deferred by default -- run the router evaluation first).")
    parser.add_argument('--no-wandb', action='store_true', help="Disable wandb logging for this run.")
    parser.add_argument('--wandb-run-id', type=str, default=None,
                         help="Resume and log into this existing wandb run id (e.g. the checkpoint's own "
                              "training run) instead of creating a new one.")
    cmd_args = parser.parse_args()

    config_path = cmd_args.config
    if config_path is None:
        derived = Path(cmd_args.checkpoint).resolve().parent.parent / "learn.yaml"
        if not derived.exists():
            raise FileNotFoundError(
                f"--config not given and no sibling learn.yaml found at {derived}. "
                f"Pass --config explicitly."
            )
        config_path = str(derived)
        print(f"[*] --config not given, using: {config_path}")

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    args = argparse.Namespace(**yaml_config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb = not cmd_args.no_wandb
    if use_wandb and cmd_args.wandb_run_id:
        wandb.init(project="MoE", id=cmd_args.wandb_run_id, resume="must")
    elif use_wandb:
        wandb.init(
            project="MoE",
            name=f"router_eval_{Path(cmd_args.checkpoint).parent.parent.name}_{Path(cmd_args.checkpoint).stem}",
            tags=['router_eval', getattr(args, 'dataset', 'unknown')],
            config={**yaml_config, "checkpoint": str(cmd_args.checkpoint)},
            settings=_wandb_settings(),
        )

    print(f"[*] Loading dataset: {args.dataset}")
    
    # init dataset.
    if args.dataset == 'pacs':
        full_dataset = PACSDataset(root_dir=args.data_dir, transform=None)
        num_classes = 7
    elif args.dataset == 'officehome':
        full_dataset = OfficeHomeDataset(root_dir=args.data_dir, transform=None)
        num_classes = 65
    elif args.dataset == 'cifar100':
        full_dataset = CIFAR100Dataset(root_dir=args.data_dir, split="train", transform=None)
        num_classes = 100
    elif args.dataset == 'tiny_imagenet':
        full_dataset = TinyImageNetDataset(root_dir=args.data_dir, transform=None)
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # partition based on train/test/unseen splits.
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, _ = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    unlearn_setting = getattr(args, 'unlearn_setting', 'random')
    if unlearn_setting == 'random':
        forget_size = int(getattr(args, 'forget_ratio', 0.1) * train_size)
        forget_subset, retain_subset = random_split(
            train_subset, [forget_size, train_size - forget_size], generator=generator
        )
        final_test_subset = test_subset
    elif unlearn_setting in ['class', 'domain']:
        target_list = getattr(args, f'forget_{unlearn_setting}es', [0])
        if not isinstance(target_list, list): 
            target_list = [target_list]
        
        f_tr_idx, r_tr_idx, r_te_idx = [], [], []
        for idx in train_subset.indices:
            val = full_dataset.labels[idx] if unlearn_setting == 'class' else get_domain(full_dataset, idx)
            (f_tr_idx if val in target_list else r_tr_idx).append(idx)
            
        for idx in test_subset.indices:
            val = full_dataset.labels[idx] if unlearn_setting == 'class' else get_domain(full_dataset, idx)
            if val not in target_list: 
                r_te_idx.append(idx)
                
        forget_subset = Subset(full_dataset, f_tr_idx)
        retain_subset = Subset(full_dataset, r_tr_idx)
        final_test_subset = Subset(full_dataset, r_te_idx)

    train_loader = DataLoader(ApplyTransform(train_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ApplyTransform(final_test_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)
    forget_loader = DataLoader(ApplyTransform(forget_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)
    retain_loader = DataLoader(ApplyTransform(retain_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)

    print(f"[*] Initializing model: {args.model_name}")
    
    if 'module' in args.model_name:
        model = ModuleArchitecture(
            model_name=args.model_name, 
            num_classes=num_classes, 
            moe_layers=getattr(args, 'moe_layers', None),
            num_experts=args.num_experts,
            expert_depth=args.expert_depth,
            expert_hidden_ratio=args.expert_hidden_ratio,
            gate_k=args.gate_k,
            device=device
        )
    elif 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, device=device)
    else:
        raise NotImplementedError("Evaluation script is currently customized primarily for 'module' and 'resnet' architectures.")

    checkpoint = torch.load(cmd_args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print("\n" + "="*40)
    print("ACCURACY METRICS")
    print("="*40)
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%")
    if use_wandb:
        wandb.log({"train_accuracy": train_acc, "test_accuracy": test_acc})

    if 'module' in args.model_name:
        if args.dataset != 'pacs':
            print(f"\n[*] Router evaluation currently assumes a PACS-style (image, label, domain) "
                  f"dataloader; skipping for dataset='{args.dataset}'.")
        else:
            print("\n" + "="*70)
            print("ROUTER EVALUATION (Step 2 -- standalone, no Eq. 7 involved)")
            print("="*70)

            domain_names = full_dataset.domain_names
            class_names = full_dataset.class_names
            forget_domain_idx = domain_names.index('art_painting') if 'art_painting' in domain_names else 0

            print("\n" + "-"*70)
            print("-- TRAIN split --")
            print("-"*70)
            diag_train = extract_router_diagnostics(
                model=model,
                dataloader=train_loader,
                device=device,
                num_experts=args.num_experts,
                gate_k=args.gate_k,
                num_classes=len(class_names),
                num_domains=len(domain_names),
            )
            print_router_diagnostics(
                diag_train,
                num_experts=args.num_experts,
                gate_k=args.gate_k,
                domain_names=domain_names,
                class_names=class_names,
                forget_domain_idx=forget_domain_idx,
                log_to_wandb=use_wandb,
                split_label="TRAIN",
            )

            print("\n" + "-"*70)
            print("-- TEST split (held-out) --")
            print("-"*70)
            diag_test = extract_router_diagnostics(
                model=model,
                dataloader=test_loader,
                device=device,
                num_experts=args.num_experts,
                gate_k=args.gate_k,
                num_classes=len(class_names),
                num_domains=len(domain_names),
            )
            print_router_diagnostics(
                diag_test,
                num_experts=args.num_experts,
                gate_k=args.gate_k,
                domain_names=domain_names,
                class_names=class_names,
                forget_domain_idx=forget_domain_idx,
                log_to_wandb=use_wandb,
                split_label="TEST",
            )

        if cmd_args.run_eq7_diagnostics:
            print("\n" + "="*40)
            print("LOCALIZATION DIAGNOSTICS (MoDULE, Eq. 7 -- Step 3)")
            print("="*40)

            k_u = getattr(args, 'k_u', 2)
            alpha = getattr(args, 'alpha', 1.0)

            FEM, ARR, RFO = compute_localization_diagnostics(
                model=model,
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                device=device,
                num_experts=args.num_experts,
                gate_k=args.gate_k,
                k_u=k_u,
                alpha=alpha
            )

            print(f"Forget-expert routing mass (FEM):    {FEM:.4f}")
            print(f"At-risk retain ratio (ARR):          {ARR:.4f}")
            print(f"Retain-forget routing overlap (RFO): {RFO:.4f}")
            if use_wandb:
                wandb.log({"eq7/FEM": FEM, "eq7/ARR": ARR, "eq7/RFO": RFO})
        else:
            print("\n[*] Skipping Eq. 7 localization diagnostics (pass --run-eq7-diagnostics to enable). "
                  "Read the router evaluation above first.")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()