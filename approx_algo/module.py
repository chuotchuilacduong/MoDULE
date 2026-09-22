import os
import math
import time
import copy
import torch
import torch.nn.functional as F
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent
import inspect

from metric.fa import forget_acc
from metric.online_specialization import OnlineSpecTracker
from metric.route_separation import loss_route_separation
from module_diagnostics import (
    extract_router_diagnostics,
    print_router_diagnostics,
    compute_localization_diagnostics,
)


class Module(Gradient_Ascent):
    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            unseen_loader,
            forget_loader,
            forget_test_loader,
            retain_loader,
            retain_test_loader,
            optimizer,
            criteria,
            num_epoch,
            # config for learn
            lambda_sparse=1.0,
            lambda_balance=1.0,
            lambda_div=1.0,
            # config for unlearn
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            eta=1.0,
            k_u=2,
            # ablation config
            selection_option="diff",
            update_scope="selected_experts_and_head",
            # S12 / Table 15: công tắc TƯỜNG MINH cho ước lượng cân bằng.
            # Trước đây chọn ngầm bằng `use_ema = train_loader.batch_size <= 8`, nên với
            # batch_size=128 của mọi config thì nhánh EMA là code chết và ema_alpha vô tác dụng.
            # Đổi batch size để bật EMA sẽ làm nhiễu ablation (kích thước batch cũng đổi theo).
            # Giá trị: "minibatch" | "ema" | "none".
            balance_estimator="minibatch",
            # S13 / Table 16: "none" | "cosine" | "orthogonality" | "cka" | "output_decorrelation"
            diversity_objective="output_decorrelation",
            # L_sp tính trên tensor nào (docs/review_online_spec_and_route_separation.md, mục 2):
            #   "gate"   -- entropy của gate top-k `last_pi` [N,k] (hành vi gốc; trần log k,
            #               bị nén bởi double-softmax nếu gate_norm="softmax")
            #   "pi_all" -- H(pi)/log M trên softmax đầy đủ `last_pi_all` [N,M], paper Eq. (3)/(32)
            sparse_target="gate",
            # Probe cho CKA (Eq. 37): số mẫu token lấy ngẫu nhiên mỗi batch để tính L_div.
            # 0 = dùng toàn bộ token (hành vi gốc). Chỉ có tác dụng với diversity_objective="cka".
            probe_size=0,
            # L_sep = -I(G;E): tách tuyến đường theo nhóm (metric/route_separation.py).
            #   lambda_sep=0 -> tắt hẳn (mặc định, không đổi hành vi cũ)
            #   sep_axis: trục nhóm G. Nguyên tố: "domain" | "class" | "joint" (ô class x domain, G = C*D).
            #             Ghép bằng "+", vd "joint+domain+class"; "both" = "domain+class". List theo từng
            #             MoE layer, vd ["domain","class"] (layer 10 domain, layer 11 class); "none" = tắt layer đó.
            #             "joint": mỗi expert nhận một NHÓM Ô (vài class x vài domain) thay vì 1 class hoặc 1 domain;
            #             thêm marginal "+domain+class" để các ô cùng expert nằm cùng hàng/cột (khối chữ nhật).
            #   sep_use_gated: True -> gated mass (Eq. 29), False -> softmax thô (Mod-Squad)
            #   sep_ema_alpha: >0 để làm mượt W qua các batch khi mỗi batch có ít mẫu/nhóm
            lambda_sep=0.0,
            sep_axis="domain",
            sep_use_gated=True,
            sep_ema_alpha=0.0,
            # online_spec/* (metric/online_specialization.py): 0 = tắt, n = log mỗi n epoch
            online_spec_log_every=1,
            # Table 15 (muc H): regularizer giu tinh modular TRONG pha unlearn.
            #   "none"             -- tat han
            #   "option2_cross"    -- decorrelate selected vs frozen tren minibatch HIEN TAI
            #                         (co chua mau forget) = DUNG hanh vi code cu
            #   "option3_retained" -- nhu tren nhung tinh tren mau RETAIN, dung nhu paper mo ta
            #   "option1_geometry" -- giu ma tran tuong dong cheo giong TRUOC khi unlearn
            unlearn_modularity_reg="option2_cross",
            device="cuda",
            # optional: domain/class names for periodic router diagnostics (PACS/OfficeHome only --
            # datasets without a domain field simply skip this, see `run_full_router_diag` in learn()).
            domain_names=None,
            class_names=None,
            domain_mass_log_every=10,
            dead_expert_threshold=0.01,
            run_eq7_diagnostics=False,
            # per-epoch router train/test consistency check (see
            # metric/router_traintest_match.py). Loaders are keyed by numeric
            # domain id; empty dicts (default) disable the check.
            per_domain_train_loaders_eval=None,
            per_domain_test_loaders=None,
            # tương tự nhưng theo NHÃN LỚP, cho class_mass và router_match_class
            per_class_train_loaders_eval=None,
            per_class_test_loaders=None,
            router_match_log_every=1,
            router_match_k_u=1,
            # learn()-phase stability. defaults are off so existing runs keep
            # their exact behaviour; learn.py/retrain_baseline.py opt in via yaml.
            grad_clip_norm=None,
            lr_schedule=None,
            warmup_epochs=1,
            # micro-batches accumulated per optimizer step. 1 = original behaviour.
            # batch_size x grad_accum_steps is the effective batch, so a large-M model
            # can keep the same effective batch on a smaller activation footprint.
            grad_accum_steps=1,
            # experts active per input during the unlearning forward pass. None
            # keeps active-k tied to k_u (the original behaviour).
            unlearn_active_k=None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            unseen_loader=unseen_loader,
            forget_loader=forget_loader,
            forget_test_loader=forget_test_loader,
            retain_loader=retain_loader,
            retain_test_loader=retain_test_loader,
            optimizer=optimizer,
            criteria=criteria,
            num_epoch=num_epoch,
            device=device
        )

        # verify architecture compatibility
        actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        supported_models = ['ModuleArchitecture']
        if actual_model.__class__.__name__ not in supported_models:
            raise TypeError(f"Module does not support {self.model.__class__.__name__}. Supported: {supported_models}")

        self.lambda_sparse = lambda_sparse
        self.lambda_balance = lambda_balance
        self.lambda_div = lambda_div

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.k_u = k_u

        self.selection_option = selection_option
        self.update_scope = update_scope
        if balance_estimator not in ("minibatch", "ema", "none"):
            raise ValueError(
                f"balance_estimator must be 'minibatch', 'ema' or 'none' (got {balance_estimator!r})")
        self.balance_estimator = balance_estimator
        if diversity_objective not in ("none","cosine","orthogonality","cka","output_decorrelation"):
            raise ValueError(f"diversity_objective không hợp lệ: {diversity_objective!r}")
        self.diversity_objective = diversity_objective
        if sparse_target not in ("gate", "pi_all"):
            raise ValueError(f"sparse_target phải là 'gate' hoặc 'pi_all' (nhận {sparse_target!r})")
        self.sparse_target = sparse_target
        self.probe_size = int(probe_size)
        def _parse_axis(a):
            a = "domain+class" if a == "both" else str(a)
            atoms = [t.strip() for t in a.split("+") if t.strip()]
            bad = [t for t in atoms if t not in ("domain", "class", "joint", "none")]
            if bad:
                raise ValueError(f"sep_axis chỉ nhận domain/class/joint/none ghép bằng '+' (nhận {bad})")
            return [t for t in atoms if t != "none"]
        if isinstance(sep_axis, (list, tuple)):
            sep_axis = [_parse_axis(a) for a in sep_axis]      # list[list[str]] theo layer
        else:
            sep_axis = _parse_axis(sep_axis)                   # list[str] cho mọi layer
        self.lambda_sep = float(lambda_sep)
        self.sep_axis = sep_axis
        self.sep_use_gated = bool(sep_use_gated)
        self.sep_ema_alpha = float(sep_ema_alpha)
        self.online_spec_log_every = int(online_spec_log_every)
        if unlearn_modularity_reg not in ("none","option1_geometry","option2_cross","option3_retained"):
            raise ValueError(f"unlearn_modularity_reg khong hop le: {unlearn_modularity_reg!r}")
        self.unlearn_modularity_reg = unlearn_modularity_reg

        self.domain_names = domain_names
        self.class_names = class_names
        self.domain_mass_log_every = domain_mass_log_every
        self.dead_expert_threshold = dead_expert_threshold
        self.run_eq7_diagnostics = run_eq7_diagnostics

        self.per_domain_train_loaders_eval = per_domain_train_loaders_eval or {}
        self.per_domain_test_loaders = per_domain_test_loaders or {}
        self.per_class_train_loaders_eval = per_class_train_loaders_eval or {}
        self.per_class_test_loaders = per_class_test_loaders or {}
        self.router_match_log_every = router_match_log_every
        self.router_match_k_u = router_match_k_u
        self.grad_clip_norm = grad_clip_norm
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        # đánh giá FA/RA/TA/MIA mỗi N epoch trong learn(). 0 = tắt (mặc định,
        # giống hành vi cũ: chỉ đánh giá một lần sau vòng lặp). Đặt 1 để lấy
        # quỹ đạo FA theo epoch, ví dụ cho baseline Retraining.
        self.learn_eval_every = 0
        self.unlearn_active_k = unlearn_active_k

        # full router diagnostics (entropy, dead experts, per-domain mass, per-expert
        # specialization) need a (image, label, domain) dataloader and class names --
        # only available for domain-labeled datasets like PACS/OfficeHome. Computed once
        # here so both learn() and unlearn() can reuse it without recomputing.
        moe_layers_ref = [m for m in self.model.modules() if m.__class__.__name__ == 'DeepMoELayer']
        self.num_experts = moe_layers_ref[0].num_experts if moe_layers_ref else 0
        self.gate_k = moe_layers_ref[0].gate_k if moe_layers_ref else 0
        self.run_full_router_diag = bool(self.domain_names) and bool(self.class_names) and self.num_experts > 0
        self.forget_domain_idx = (
            self.domain_names.index('art_painting')
            if self.run_full_router_diag and 'art_painting' in self.domain_names else 0
        )

    def _run_final_router_diagnostics(self, phase):
        """Runs once (not periodically -- each split below is a full extra pass over its
        loader, and Eq.7 adds two more, so this is only worth paying for at the end of
        learn()/unlearn(), not every epoch). Reports TRAIN and TEST separately so you can
        see whether routing learned on TRAIN still holds on held-out TEST images, instead
        of only ever inspecting one split."""
        if not self.run_full_router_diag:
            return

        for split_name, loader in [("TRAIN", self.train_loader), ("TEST", self.test_loader)]:
            if loader is None:
                continue
            split_label = f"{phase}_{split_name}"
            print(f"\n[*] Running full router diagnostics on {split_label} split...")
            diag = extract_router_diagnostics(
                model=self.model, dataloader=loader, device=self.device,
                num_experts=self.num_experts, gate_k=self.gate_k,
                num_classes=len(self.class_names), num_domains=len(self.domain_names),
            )
            print_router_diagnostics(
                diag, num_experts=self.num_experts, gate_k=self.gate_k,
                domain_names=self.domain_names, class_names=self.class_names,
                forget_domain_idx=self.forget_domain_idx, dead_expert_threshold=self.dead_expert_threshold,
                log_to_wandb=True, split_label=split_label,
            )

        if self.run_eq7_diagnostics:
            print(f"\n[*] Running Eq.7 localization diagnostics (FEM/ARR/RFO) [{phase}]...")
            FEM, ARR, RFO = compute_localization_diagnostics(
                model=self.model, forget_loader=self.forget_loader, retain_loader=self.retain_loader,
                device=self.device, num_experts=self.num_experts, gate_k=self.gate_k,
                k_u=self.k_u, alpha=self.alpha,
            )
            print(f"Forget-expert routing mass (FEM):    {FEM:.4f}")
            print(f"At-risk retain ratio (ARR):          {ARR:.4f}")
            print(f"Retain-forget routing overlap (RFO): {RFO:.4f}")
            wandb.log({
                f"eq7_{phase.lower()}/FEM": FEM,
                f"eq7_{phase.lower()}/ARR": ARR,
                f"eq7_{phase.lower()}/RFO": RFO,
            })

    @torch.no_grad()
    def _mass_on_loader(self, loader, num_classes, num_domains):
        """Tính ma trận (lớp x expert) và (domain x expert) trên một loader cho trước.

        Dùng cho tập TEST: các khoá class_mass/domain_mass không hậu tố được tích luỹ
        ngay trong vòng lặp huấn luyện nên chúng thuộc tập TRAIN và có augmentation.
        Hàm này chạy một lượt forward riêng với transform tất định.
        """
        was_training = self.model.training
        self.model.eval()
        cmass, cnt_c, dmass, cnt_d = {}, None, {}, None
        for batch in loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device).long()
            domains = batch[2].to(self.device).long() if len(batch) > 2 else None
            self.model.inference(images)
            for name, module in self.model.featurizer.model.named_modules():
                if module.__class__.__name__ != 'DeepMoELayer':
                    continue
                pi = module.last_pi_all.detach()
                E = pi.size(-1)
                S = pi.size(0) // labels.size(0)
                if name not in cmass:
                    cmass[name] = torch.zeros(num_classes, E, device=self.device)
                    dmass[name] = torch.zeros(num_domains, E, device=self.device)
                cmass[name].index_add_(0, labels.repeat_interleave(S), pi)
                if domains is not None:
                    dmass[name].index_add_(0, domains.repeat_interleave(S), pi)
            if cnt_c is None:
                cnt_c = torch.zeros(num_classes, device=self.device)
                cnt_d = torch.zeros(num_domains, device=self.device)
            cnt_c.index_add_(0, labels, torch.full((labels.size(0),), float(S), device=self.device))
            if domains is not None:
                cnt_d.index_add_(0, domains, torch.full((domains.size(0),), float(S), device=self.device))
        if was_training:
            self.model.train()
        return cmass, cnt_c, dmass, cnt_d

    def _loss_sparse(self, pi):
        """sparse_target="gate": entropy thô của gate top-k (gốc). sparse_target="pi_all":
        H(pi)/log M trên softmax đầy đủ, paper Eq. (3)/(32), nằm trong [0, 1]."""
        entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
        if getattr(self, "sparse_target", "gate") == "pi_all":
            M = pi.size(-1)
            return entropy.mean() / math.log(M) if M > 1 else entropy.mean()
        return entropy.mean()

    def _loss_balance(self, gate_mass, module_name, use_ema, ema_states, ema_alpha):
        """Gated-mass balancing, paper Eq. (4)/(34).

        L_bal = M * sum_m (gbar_m - 1/M)^2,  gbar_m = mean_tokens g_{t,m}

        `gate_mass` phải là [N, M] theo DANH TÍNH expert (DeepMoELayer.last_gate_mass).
        Trước đây hàm này nhận `last_pi` có shape [N, k], nên `M = pi.size(-1)` đọc ra
        k chứ không phải M, và nó cân bằng THỨ HẠNG gate thay vì cân bằng expert --
        tức không hề có áp lực chống sụp đổ router. Hệ số M đứng trước cũng bị thiếu.
        """
        M = gate_mass.size(-1)
        mean_mass = gate_mass.mean(dim=0)

        if use_ema:
            if module_name not in ema_states:
                ema_states[module_name] = torch.ones_like(mean_mass) / M
            effective = ema_alpha * ema_states[module_name] + (1 - ema_alpha) * mean_mass
            ema_states[module_name] = effective.detach()
        else:
            effective = mean_mass

        return M * ((effective - 1.0 / M) ** 2).sum()

    # ------------------------------------------------------------------ #
    # S13 / Table 16: năm mục tiêu phân hoá expert.
    #
    # Lưu ý sai lệch code-paper: paper Eq. (5)/(39) mô tả L_div là CENTERED
    # linear CKA, nhưng hàm duy nhất có trong repo lại là phạt bình phương
    # Frobenius của tương quan chéo, KHÔNG centering và chuẩn hoá bằng ||H||_F
    # thay vì ||H^T H||_F. Đó là output decorrelation, không phải CKA. Nên hàm
    # cũ được giữ nguyên hành vi dưới tên "output_decorrelation" (mọi kết quả
    # đã chạy trước đây đều thuộc mục tiêu này), và "cka" là bản cài đúng Eq. 39.
    # ------------------------------------------------------------------ #
    def _probe_subsample(self, h_stack):
        """Probe B_p token ngẫu nhiên cho L_div (paper Eq. 37). probe_size=0 -> giữ nguyên."""
        n = getattr(self, "probe_size", 0)
        if n <= 0 or h_stack is None or h_stack.size(0) <= n:
            return h_stack
        idx = torch.randperm(h_stack.size(0), device=h_stack.device)[:n]
        return h_stack[idx]

    def _loss_diversity(self, h_stack, eps=1e-6):
        obj = getattr(self, "diversity_objective", "output_decorrelation")
        if obj == "none":
            return h_stack.new_zeros(())
        if h_stack.shape[1] < 2:
            return h_stack.new_zeros(())
        fn = {
            "output_decorrelation": self._div_output_decorrelation,
            "cka": self._div_centered_cka,
            "cosine": self._div_cosine,
            "orthogonality": self._div_orthogonality,
        }.get(obj)
        if fn is None:
            raise ValueError(f"diversity_objective không hợp lệ: {obj!r}")
        return fn(h_stack, eps)

    def _div_output_decorrelation(self, h_stack, eps=1e-6):
        """Hàm gốc của repo, giữ nguyên từng phép tính để tái lập được kết quả cũ."""
        B, M, r = h_stack.shape
        loss = h_stack.new_zeros(1).squeeze()
        H_tilde = []
        for m in range(M):
            H_m = h_stack[:, m, :]
            H_tilde.append(H_m / H_m.norm(p='fro').clamp(min=eps))
        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                C = (H_tilde[m].T @ H_tilde[n])
                loss += (C ** 2).sum()
        return loss

    def _div_centered_cka(self, h_stack, eps=1e-6):
        """Centered linear CKA, paper Eq. (38)-(40).

        H~ = J H với J = I - (1/Bp) 11^T ; CKA = ||H~m^T H~n||_F^2
                                                / (||H~m^T H~m||_F ||H~n^T H~n||_F)
        Trung bình trên các cặp KHÔNG thứ tự, hệ số 2/(M(M-1)).
        """
        B, M, r = h_stack.shape
        Hc = h_stack - h_stack.mean(dim=0, keepdim=True)      # centering theo probe batch
        gram_self = [(Hc[:, m, :].T @ Hc[:, m, :]).norm(p='fro').clamp(min=eps) for m in range(M)]
        loss = h_stack.new_zeros(1).squeeze()
        for m in range(M):
            for n in range(m + 1, M):
                cross = (Hc[:, m, :].T @ Hc[:, n, :]).norm(p='fro') ** 2
                loss = loss + cross / (gram_self[m] * gram_self[n])
        return loss * (2.0 / (M * (M - 1)))

    def _div_cosine(self, h_stack, eps=1e-6):
        """Cosine similarity trung bình giữa đầu ra các expert trên từng mẫu probe."""
        B, M, r = h_stack.shape
        Hn = h_stack / h_stack.norm(dim=-1, keepdim=True).clamp(min=eps)   # [B, M, r]
        sim = torch.einsum("bmr,bnr->bmn", Hn, Hn)                          # [B, M, M]
        off = ~torch.eye(M, dtype=torch.bool, device=h_stack.device)
        return sim[:, off].abs().mean()

    def _div_orthogonality(self, h_stack, eps=1e-6):
        """||W^T W - I||_F^2 trên ma trận đầu ra expert đã chuẩn hoá cột."""
        B, M, r = h_stack.shape
        H = h_stack.permute(1, 0, 2).reshape(M, -1)                         # [M, B*r]
        H = H / H.norm(dim=-1, keepdim=True).clamp(min=eps)
        G = H @ H.T
        I = torch.eye(M, device=h_stack.device, dtype=G.dtype)
        return ((G - I) ** 2).sum() / (M * (M - 1))

    # helper function for unlearning phase.
    # get forget and retain mass.
    def _get_routing_mass(self, loader):
        self.model.eval()
        masses = None
        total_tokens = 0
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                self.model(images)

                batch_masses = []
                num_tokens = 0
                for _, m in self.model.named_modules():
                    if m.__class__.__name__ == 'DeepMoELayer':
                        batch_masses.append(m.last_pi_all.sum(dim=0))
                        num_tokens = m.last_pi_all.size(0)

                if masses is None:
                    masses = batch_masses
                else:
                    masses = [m + b for m, b in zip(masses, batch_masses)]
                total_tokens += num_tokens
        return [m / max(total_tokens, 1) for m in masses]

    def _apply_update_scope(self, selected_experts_per_layer, moe_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        if self.update_scope == "full_model":
            for param in self.model.parameters():
                param.requires_grad = True
            return

        if self.update_scope == "all_experts":
            for m in moe_layers:
                for expert in m.experts:
                    for param in expert.parameters():
                        param.requires_grad = True
            return

        for l_idx, m in enumerate(moe_layers):
            selected_experts = selected_experts_per_layer[l_idx]
            for expert_idx, expert in enumerate(m.experts):
                if expert_idx in selected_experts:
                    for param in expert.parameters():
                        param.requires_grad = True

        if self.update_scope == "selected_experts_only":
            pass  # already handled.

        elif self.update_scope == "selected_experts_and_head":
            for param in self.model.classifier_head.parameters():
                param.requires_grad = True

        elif self.update_scope == "selected_experts_and_router":
            for m in moe_layers:
                for param in m.router.parameters():
                    param.requires_grad = True

        elif self.update_scope == "selected_experts_and_last_block":
            last_block = self.model.featurizer.model.blocks[-1]
            for param in last_block.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown update_scope: {self.update_scope}")

    def _get_gradient_influence(self, data_loader):
        self.model.train()
        self.optimizer.zero_grad()

        for batch in data_loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            logits, _ = self.model.forward_with_grad(images)
            loss = self._unlearn_loss_forget(logits, labels)

            loss.backward()

        moe_layers = [m for _, m in self.model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']
        grad_scores = []

        for m in moe_layers:
            layer_scores = []
            for expert in m.experts:
                grad_mag = 0.0
                for param in expert.parameters():
                    if param.grad is not None:
                        grad_mag += param.grad.abs().sum().item()
                layer_scores.append(grad_mag)

            grad_scores.append(torch.tensor(layer_scores, device=self.device))

        self.optimizer.zero_grad()
        return grad_scores

    def _unlearn_loss_forget(self, logits_f, labels_f):
        return -self.criteria(logits_f, labels_f)

    def _unlearn_loss_retain(self, logits_r, labels_r):
        return self.criteria(logits_r, labels_r)

    def _unlearn_loss_distill(self, logits_r, images_r, origin_model):
        with torch.no_grad():
            orig_logits_r, _ = origin_model.forward_with_grad(images_r)

        log_preds = F.log_softmax(logits_r, dim=-1)
        target_preds = F.softmax(orig_logits_r, dim=-1)
        return F.kl_div(log_preds, target_preds, reduction='batchmean')

    def _sep_from_H(self, H, m, selected_M_f):
        """Phat tuong quan cheo giua expert DUOC CHON va expert BI DONG BANG, tren H cho truoc."""
        loss = torch.tensor(0.0, device=self.device)
        frozen = [i for i in range(m.num_experts) if i not in selected_M_f]
        for em in selected_M_f:
            for n in frozen:
                Hm = H[:, em, :]; Hn = H[:, n, :]
                Hm_t = Hm / (torch.norm(Hm, p='fro') + 1e-8)
                Hn_t = Hn / (torch.norm(Hn, p='fro') + 1e-8)
                loss = loss + torch.norm(torch.matmul(Hm_t.t(), Hn_t), p='fro') ** 2
        return loss

    def _sim_matrix(self, H, eps=1e-8):
        """Ma tran tuong dong cheo giua moi cap expert: [M, M]."""
        M = H.size(1)
        Hn = H / (H.flatten(0, 0).norm(dim=(0, 2), keepdim=True).transpose(0, 1) + eps) \
             if False else H / (H.norm(dim=(0, 2), keepdim=True) + eps)
        return torch.einsum("bmr,bnr->mn", Hn, Hn)

    def _unlearn_loss_geometry(self, moe_layers, H0_per_layer):
        """Option 1: giu nguyen ma tran tuong dong cheo da hoc TRUOC khi unlearn."""
        loss = torch.tensor(0.0, device=self.device)
        for l_idx, m in enumerate(moe_layers):
            H0 = H0_per_layer.get(l_idx)
            if H0 is None:
                continue
            loss = loss + ((self._sim_matrix(m.last_h) - self._sim_matrix(H0)) ** 2).sum()
        return loss

    def _unlearn_loss_separation(self, moe_layers, selected_experts_per_layer):
        loss_sep = torch.tensor(0.0, device=self.device)
        for l_idx, m in enumerate(moe_layers):
            H = m.last_h
            batch_tokens_len = H.size(0)

            selected_M_f = selected_experts_per_layer[l_idx]
            frozen_M_r = [i for i in range(m.num_experts) if i not in selected_M_f]

            for expert_m in selected_M_f:
                for n in frozen_M_r:
                    Hm = H[:, expert_m, :]
                    Hn = H[:, n, :]

                    norm_m = torch.norm(Hm, p='fro') + 1e-8
                    norm_n = torch.norm(Hn, p='fro') + 1e-8

                    Hm_tilde = Hm / norm_m
                    Hn_tilde = Hn / norm_n

                    # old version -> vanishing.
                    # inner_product = torch.matmul(Hm_tilde.t(), Hn_tilde) / batch_tokens_len

                    # new version.
                    inner_product = torch.matmul(Hm_tilde.t(), Hn_tilde)
                    loss_sep += torch.norm(inner_product, p='fro') ** 2
        return loss_sep

    def learn(self, ckpt_path, ema_alpha=0.9):
        self.model._set_grad_mode("learning")
        use_ema = (self.balance_estimator == "ema")
        print(f"[*] balance estimator: {self.balance_estimator}"
              + (" (L_bal bị tắt hoàn toàn)" if self.balance_estimator == "none" else ""))
        ema_states = {}
        sep_ema_states = {}
        print(f"[*] sparse target: {self.sparse_target} | diversity: {self.diversity_objective}"
              f" (probe_size={self.probe_size}) | lambda_sep={self.lambda_sep} axis={self.sep_axis}"
              f" gated={self.sep_use_gated} ema={self.sep_ema_alpha} | online_spec every {self.online_spec_log_every}")
        total_train_time = 0.0
        total_train_steps = 0
        best_loss = float('inf')
        best_ckpt_path = f"{ckpt_path}_best.pt"

        num_domains = len(self.domain_names) if self.domain_names else 0

        # cosine decay with linear warmup, stepped per batch. paired with the
        # gradient clipping below: a constant LR for the whole run is what let
        # the base model diverge late in training.
        scheduler = None
        if self.lr_schedule == "cosine":
            # count OPTIMIZER steps, not micro-batches, so the schedule is identical
            # whether or not gradient accumulation is in use.
            steps_per_epoch = max(math.ceil(len(self.train_loader) / self.grad_accum_steps), 1)
            total_steps = steps_per_epoch * self.num_epoch
            warmup_steps = min(steps_per_epoch * self.warmup_epochs, max(total_steps - 1, 0))

            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            print(f"[*] lr schedule: cosine over {total_steps} steps "
                  f"({self.warmup_epochs} warmup epoch(s)) | grad clip: {self.grad_clip_norm}")

        for epoch in range(self.num_epoch):
            self.model.train()
            epoch_start_time = time.time()

            running_total, running_ce, running_sp, running_bal, running_div = 0.0, 0.0, 0.0, 0.0, 0.0
            running_sep = 0.0

            # online_spec/*: thống kê chuyên biệt hoá từng epoch, tái dùng pi đã cache trong
            # forward pass (không tốn inference thêm). Tạo MỚI mỗi epoch -- tracker không có reset().
            do_spec_log = self.online_spec_log_every > 0 and (
                    (epoch + 1) % self.online_spec_log_every == 0 or epoch == self.num_epoch - 1)
            spec_tracker = None
            if do_spec_log:
                spec_tracker = OnlineSpecTracker(
                    self.num_experts, self.gate_k,
                    num_domains=num_domains,
                    num_classes=len(self.class_names) if self.class_names else 0,
                    dead_expert_threshold=self.dead_expert_threshold,
                )

            # domain-mass tracking: only active every `domain_mass_log_every` epochs (and the
            # last epoch), reusing the pi already computed by the forward pass -- no extra
            # inference cost. Skipped entirely for datasets without a domain field.
            do_domain_log = num_domains > 0 and (
                    (epoch + 1) % self.domain_mass_log_every == 0 or epoch == self.num_epoch - 1
            )
            domain_mass_sum, domain_tok_count = {}, {}
            class_mass_sum, class_tok_count = {}, {}

            for micro_step, batch in enumerate(self.train_loader):
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                batch_domains = batch[2].to(self.device).long() if len(batch) > 2 else None
                domains = batch_domains if do_domain_log else None
                labels_for_mass = labels if do_domain_log else None

                # gradient accumulation: zero only at the start of an accumulation
                # window, step only at its end. with grad_accum_steps == 1 this is
                # exactly the original per-batch behaviour.
                if micro_step % self.grad_accum_steps == 0:
                    self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)

                all_pi, all_h, all_gate_mass, moe_names, moe_modules = [], [], [], [], []
                for name, module in self.model.featurizer.model.named_modules():
                    if module.__class__.__name__ == 'DeepMoELayer':
                        all_pi.append(module.last_pi_all if self.sparse_target == "pi_all" else module.last_pi)
                        all_h.append(self._probe_subsample(module.last_h))
                        all_gate_mass.append(module.last_gate_mass)
                        moe_names.append(name)
                        moe_modules.append((name, module))

                        if domains is not None:
                            pi_all = module.last_pi_all.detach()  # [B*S, num_experts]
                            num_experts = pi_all.size(-1)
                            S = pi_all.size(0) // domains.size(0)
                            domain_tok = domains.repeat_interleave(S)
                            if name not in domain_mass_sum:
                                domain_mass_sum[name] = torch.zeros(num_domains, num_experts, device=self.device)
                                domain_tok_count[name] = torch.zeros(num_domains, device=self.device)
                            domain_mass_sum[name].index_add_(0, domain_tok, pi_all)
                            domain_tok_count[name].index_add_(0, domain_tok,
                                                              torch.ones_like(domain_tok, dtype=torch.float))

                        if labels_for_mass is not None:
                            pi_all_c = module.last_pi_all.detach()
                            n_exp = pi_all_c.size(-1)
                            Sc = pi_all_c.size(0) // labels_for_mass.size(0)
                            cls_tok = labels_for_mass.repeat_interleave(Sc)
                            n_cls = len(self.class_names) if self.class_names else int(cls_tok.max().item()) + 1
                            if name not in class_mass_sum:
                                class_mass_sum[name] = torch.zeros(n_cls, n_exp, device=self.device)
                                class_tok_count[name] = torch.zeros(n_cls, device=self.device)
                            class_mass_sum[name].index_add_(0, cls_tok, pi_all_c)
                            class_tok_count[name].index_add_(0, cls_tok,
                                                             torch.ones_like(cls_tok, dtype=torch.float))

                ce_loss = self.criteria(logits, labels)

                if len(all_pi) > 0:
                    sp_loss = sum([self._loss_sparse(pi) for pi in all_pi]) / len(all_pi)
                    if self.balance_estimator == "none":
                        bal_loss = torch.tensor(0.0, device=self.device)
                    else:
                        bal_loss = sum([self._loss_balance(gm, n, use_ema, ema_states, ema_alpha) for gm, n in
                                        zip(all_gate_mass, moe_names)]) / len(all_gate_mass)
                    div_loss = sum([self._loss_diversity(h) for h in all_h]) / len(all_h)
                else:
                    sp_loss = torch.tensor(0.0, device=self.device)
                    bal_loss = torch.tensor(0.0, device=self.device)
                    div_loss = torch.tensor(0.0, device=self.device)

                sep_loss = torch.tensor(0.0, device=self.device)
                if self.lambda_sep > 0 and len(moe_modules) > 0:
                    n_cls = len(self.class_names) if self.class_names else int(self.model.num_classes)
                    joint = (labels * num_domains + batch_domains) if (batch_domains is not None and num_domains > 0) else None
                    group_of = {"domain": (batch_domains, num_domains), "class": (labels, n_cls),
                                "joint": (joint, n_cls * max(num_domains, 1))}
                    # trục theo layer: list[str] -> áp cho mọi layer; list[list] -> theo thứ tự MoE layer
                    if self.sep_axis and isinstance(self.sep_axis[0], list):
                        axes = [self.sep_axis[i] if i < len(self.sep_axis) else []
                                for i in range(len(moe_modules))]
                    else:
                        axes = [self.sep_axis] * len(moe_modules)
                    terms = []
                    for (name, module), ax in zip(moe_modules, axes):
                        for a in ax:
                            g, n_g = group_of[a]
                            if g is None or n_g < 2:
                                continue
                            _sep = loss_route_separation(
                                [(name, module)], g, n_g, use_gated=self.sep_use_gated,
                                ema_states=sep_ema_states.setdefault(a, {}), ema_alpha=self.sep_ema_alpha)
                            if _sep is not None:
                                terms.append(_sep)
                    if terms:
                        sep_loss = sum(terms) / len(terms)

                t_loss = ce_loss + (self.lambda_sparse * sp_loss) + (self.lambda_balance * bal_loss) + (
                            self.lambda_div * div_loss) + (self.lambda_sep * sep_loss)

                # scale so the accumulated gradient equals the mean over the full
                # effective batch, not its sum. exact for this model (LayerNorm
                # only, no BatchNorm), so micro-batch x accum == one large batch.
                (t_loss / self.grad_accum_steps).backward()

                is_last_micro = (micro_step % self.grad_accum_steps == self.grad_accum_steps - 1) or \
                                (micro_step == len(self.train_loader) - 1)
                if is_last_micro:
                    # constant-LR AdamW on a pretrained backbone blew this run up at
                    # epoch ~57 of the M=12/k=4 base model (train CE 0.056 -> 1.81,
                    # never recovered). clip before stepping so a single bad batch
                    # cannot destroy a converged model.
                    if self.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                if spec_tracker is not None:
                    spec_tracker.update(self.model, labels, batch_domains)

                running_total += t_loss.item()
                running_ce += ce_loss.item()
                running_sp += sp_loss.item()
                running_bal += bal_loss.item()
                running_div += div_loss.item()
                running_sep += sep_loss.item()

                # count optimizer steps so step totals stay comparable across runs
                # with different grad_accum_steps.
                if is_last_micro:
                    total_train_steps += 1

            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time

            num_batches = len(self.train_loader)
            avg_loss = running_total / num_batches

            print(f"epoch [{epoch + 1}/{self.num_epoch}] | "
                  f"total_loss: {avg_loss:.4f} (ce: {running_ce / num_batches:.4f}, sp: {running_sp / num_batches:.4f}, bal: {running_bal / num_batches:.4f}, div: {running_div / num_batches:.4f}, sep: {running_sep / num_batches:.4f}) | "
                  f"steps in epoch: {num_batches} | total_steps: {total_train_steps} | time: {epoch_train_time:.2f}s")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "ce_loss": running_ce / num_batches,
                "sp_loss": running_sp / num_batches,
                "bal_loss": running_bal / num_batches,
                "div_loss": running_div / num_batches,
                "sp_loss_weighted": self.lambda_sparse * running_sp / num_batches,
                "bal_loss_weighted": self.lambda_balance * running_bal / num_batches,
                "div_loss_weighted": self.lambda_div * running_div / num_batches,
                "sep_loss": running_sep / num_batches,
                "sep_loss_weighted": self.lambda_sep * running_sep / num_batches,
                "train_steps_accum": total_train_steps
            })

            if spec_tracker is not None:
                spec_tracker.log(epoch + 1)

            if do_domain_log and domain_mass_sum:
                # keep the dashboard readable: one small scalar per layer for the
                # line-chart panels (how much routing mass differs across domains,
                # averaged over experts), and the full domain x expert breakdown as
                # a Table -- tables live in their own tab and don't clutter the
                # scalar charts, and this whole block only runs every
                # `domain_mass_log_every` epochs.
                domain_payload = {"epoch": epoch + 1}
                for name, mass_sum in domain_mass_sum.items():
                    mass = mass_sum / domain_tok_count[name].clamp(min=1).unsqueeze(-1)  # [D, E]
                    spread = (mass.max(dim=0).values - mass.min(dim=0).values).mean().item()
                    domain_payload[f"domain_mass/{name}/spread"] = spread
                    domain_payload[f"domain_mass/{name}/table"] = wandb.Table(
                        columns=["domain"] + [f"e{e}" for e in range(mass.size(1))],
                        data=[[self.domain_names[d]] + mass[d].tolist() for d in range(mass.size(0))],
                    )
                    shrd = mass.sum(dim=0)
                    domain_payload[f"domain_mass/{name}/collapse"] = (
                        shrd.max() / shrd.sum().clamp(min=1e-12)).item()
                    for d in range(mass.size(0)):
                        for e in range(mass.size(1)):
                            domain_payload[f"domain_mass/{name}/{self.domain_names[d]}/e{e}"] = mass[d, e].item()
                wandb.log(domain_payload)
                if class_mass_sum:
                    class_payload = {"epoch": epoch + 1}
                    for name, mass_sum in class_mass_sum.items():
                        massc = mass_sum / class_tok_count[name].clamp(min=1).unsqueeze(-1)
                        class_payload[f"class_mass/{name}/spread"] = (
                            massc.max(dim=0).values - massc.min(dim=0).values).mean().item()
                        cnames = self.class_names or [str(i) for i in range(massc.size(0))]
                        class_payload[f"class_mass/{name}/table"] = wandb.Table(
                            columns=["class"] + [f"e{e}" for e in range(massc.size(1))],
                            data=[[cnames[c]] + massc[c].tolist() for c in range(massc.size(0))],
                        )
                        # scalar phẳng: wandb.Table chỉ cho một ảnh chụp (vẽ ra biểu đồ cột),
                        # còn các khoá dưới đây vẽ được thành ĐƯỜNG theo epoch.
                        shr = massc.sum(dim=0)
                        class_payload[f"class_mass/{name}/collapse"] = (
                            shr.max() / shr.sum().clamp(min=1e-12)).item()
                        for c in range(massc.size(0)):
                            for e in range(massc.size(1)):
                                class_payload[f"class_mass/{name}/{cnames[c]}/e{e}"] = massc[c, e].item()
                    wandb.log(class_payload)
                    print(f"  [class-mass @ epoch {epoch + 1}] " +
                          ", ".join(f"{n}: spread={v:.4f}"
                                    for n, v in class_payload.items() if n.endswith("/spread")))

                # bảng trên tập TEST (transform tất định, không augmentation)
                if self.test_loader is not None:
                    n_cls = len(self.class_names) if self.class_names else 7
                    cm_t, cc_t, dm_t, cd_t = self._mass_on_loader(self.test_loader, n_cls, num_domains)
                    test_payload = {"epoch": epoch + 1}
                    cnames_t = self.class_names or [str(i) for i in range(n_cls)]
                    for name, ms in cm_t.items():
                        Mt = ms / cc_t.clamp(min=1).unsqueeze(-1)
                        test_payload[f"class_mass_test/{name}/spread"] = (
                            Mt.max(0).values - Mt.min(0).values).mean().item()
                        sh = Mt.sum(0)
                        test_payload[f"class_mass_test/{name}/collapse"] = (
                            sh.max() / sh.sum().clamp(min=1e-12)).item()
                        test_payload[f"class_mass_test/{name}/table"] = wandb.Table(
                            columns=["class"] + [f"e{e}" for e in range(Mt.size(1))],
                            data=[[cnames_t[c]] + Mt[c].tolist() for c in range(Mt.size(0))])
                        for c in range(Mt.size(0)):
                            for e in range(Mt.size(1)):
                                test_payload[f"class_mass_test/{name}/{cnames_t[c]}/e{e}"] = Mt[c, e].item()
                    for name, ms in dm_t.items():
                        Dt = ms / cd_t.clamp(min=1).unsqueeze(-1)
                        test_payload[f"domain_mass_test/{name}/spread"] = (
                            Dt.max(0).values - Dt.min(0).values).mean().item()
                        shd = Dt.sum(0)
                        test_payload[f"domain_mass_test/{name}/collapse"] = (
                            shd.max() / shd.sum().clamp(min=1e-12)).item()
                        test_payload[f"domain_mass_test/{name}/table"] = wandb.Table(
                            columns=["domain"] + [f"e{e}" for e in range(Dt.size(1))],
                            data=[[self.domain_names[d]] + Dt[d].tolist() for d in range(Dt.size(0))])
                        for d in range(Dt.size(0)):
                            for e in range(Dt.size(1)):
                                test_payload[f"domain_mass_test/{name}/{self.domain_names[d]}/e{e}"] = Dt[d, e].item()
                    wandb.log(test_payload)
                    print(f"  [mass-TEST @ epoch {epoch + 1}] " +
                          ", ".join(f"{n.split('/')[0]}:{n.split('/')[1]}={v:.4f}"
                                    for n, v in test_payload.items() if n.endswith("/collapse")))

                print(f"  [domain-mass @ epoch {epoch + 1}] " +
                      ", ".join(f"{n}: spread={v:.4f}" for n, v in domain_payload.items() if n.endswith("/spread")))

            # per-epoch router train/test consistency: does each domain route
            # to the same top-k_u expert set on held-out test images as it does
            # on train images? runs in eval() with test transforms on both
            # sides -- see metric/router_traintest_match.py for the rationale.
            if (
                    self.per_domain_train_loaders_eval
                    and self.per_domain_test_loaders
                    and (epoch + 1) % self.router_match_log_every == 0
            ):
                from metric.router_traintest_match import domain_train_test_expert_match
                match_metrics = domain_train_test_expert_match(
                    self.model,
                    self.per_domain_train_loaders_eval,
                    self.per_domain_test_loaders,
                    self.device,
                    k_u=self.router_match_k_u,
                    domain_names=self.domain_names,
                )
                if match_metrics:
                    match_metrics["epoch"] = epoch + 1
                    wandb.log(match_metrics)
                    print(f"  [router-match @ epoch {epoch + 1}] "
                          f"mass_exact={match_metrics['router_match/overall/mass_exact_match_rate']:.4f} "
                          f"sel_exact={match_metrics['router_match/overall/sel_exact_match_rate']:.4f} "
                          f"sel_tv={match_metrics['router_match/overall/sel_mean_tv']:.4f} "
                          f"(k_u={self.router_match_k_u}, "
                          f"gate_k={match_metrics['router_match/overall/gate_k']}, "
                          f"{match_metrics['router_match/overall/num_domains']} domains)")
                # cùng phép đo nhưng nhóm theo NHÃN LỚP thay vì domain
                if self.per_class_train_loaders_eval and self.per_class_test_loaders:
                    cmatch = domain_train_test_expert_match(
                        self.model,
                        self.per_class_train_loaders_eval,
                        self.per_class_test_loaders,
                        self.device,
                        k_u=self.router_match_k_u,
                        domain_names=self.class_names,
                        prefix="router_match_class",
                        group_kind="class",
                    )
                    if cmatch:
                        cmatch["epoch"] = epoch + 1
                        wandb.log(cmatch)
                        print(f"  [router-match-CLASS @ epoch {epoch + 1}] "
                              f"mass_exact={cmatch['router_match_class/overall/mass_exact_match_rate']:.4f} "
                              f"sel_exact={cmatch['router_match_class/overall/sel_exact_match_rate']:.4f} "
                              f"sel_tv={cmatch['router_match_class/overall/sel_mean_tv']:.4f} "
                              f"({cmatch['router_match_class/overall/num_domains']} classes)")

                # `domain_train_test_expert_match` sets the model to eval();
                # restore train() before the next training epoch.
                self.model.train()

            # đánh giá theo epoch (tuỳ chọn): learn() vốn chỉ đánh giá một lần sau
            # toàn bộ vòng lặp, nên không có quỹ đạo FA/RA/TA/MIA để xem epoch nào
            # tốt hơn. Bật learn_eval_every để ghi lại từng epoch.
            if self.learn_eval_every and (epoch + 1) % self.learn_eval_every == 0:
                fa_e, ra_e, ta_e, mia_e = self.evaluate()
                self.model.train()
                print(f"  [eval @ epoch {epoch + 1}] ra: {ra_e*100:.2f}% | fa: {fa_e*100:.2f}% | "
                      f"ta: {ta_e*100:.2f}% | mia: {mia_e:.4f}")
                wandb.log({"epoch": epoch + 1, "fa": fa_e, "ra": ra_e, "ta": ta_e, "mia": mia_e})

            # keep at most one local checkpoint on disk during training (overwritten
            # in place), and push it to wandb immediately -- avoids accumulating one
            # full checkpoint per epoch on disk.
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), best_ckpt_path)
                wandb.save(best_ckpt_path, policy="now")
                print(f"[*] New best train loss {best_loss:.4f} at epoch {epoch + 1}. "
                      f"Checkpoint saved to {best_ckpt_path} and pushed to wandb.")

        print(f"[*] Training finished. Total Steps: {total_train_steps} | Running final evaluation...")
        fa_score, ra_score, ta_score, mia_score = self.evaluate()
        print(
            f"[Final Metrics] ra: {ra_score * 100:.2f}% | fa: {fa_score * 100:.2f}% | ta: {ta_score * 100:.2f}% | mia: {mia_score:.4f}")

        self._run_final_router_diagnostics(phase="LEARN")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (
                    1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_train_time_sec": total_train_time,
            "total_train_steps": total_train_steps,
            "peak_memory_gb": peak_memory_gb,
            "retain_accuracy": ra_score,
            "forget_accuracy": fa_score,
            "test_accuracy": ta_score,
            "mia_score": mia_score
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        wandb.save(f"{ckpt_path}.pt", policy="now")
        return total_train_time

    # this function for filtering retain loader.
    def _create_filtered_retain_loader(self, retain_loader, selected_experts_per_layer, moe_layers):
        self.model.eval()
        keep_indices = []
        current_idx = 0

        print("[Filter] Scanning retain set for expert intersection...")
        with torch.no_grad():
            for batch in retain_loader:
                images = batch[0].to(self.device)
                B_r = images.size(0)

                self.model(images)

                keep_mask = torch.zeros(B_r, dtype=torch.bool, device=self.device)

                for l_idx, m in enumerate(moe_layers):
                    selected_experts = selected_experts_per_layer[l_idx]
                    if not selected_experts:
                        continue

                    _, topk_indices = m.last_pi_all.topk(m.gate_k, dim=-1)
                    S = topk_indices.size(0) // B_r
                    topk_indices = topk_indices.reshape(B_r, S, -1)

                    for exp_idx in selected_experts:
                        keep_mask |= (topk_indices == exp_idx).any(dim=-1).any(dim=-1)

                true_indices = keep_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                keep_indices.extend((true_indices + current_idx).tolist())

                current_idx += B_r

        if len(keep_indices) == 0:
            print("[Filter] Warning: No retain samples matched the selected experts!")
            return None

        print(f"[Filter] Retained {len(keep_indices)} / {current_idx} samples.")

        subset = torch.utils.data.Subset(retain_loader.dataset, keep_indices)
        filtered_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=retain_loader.batch_size,
            shuffle=True,
            num_workers=retain_loader.num_workers if hasattr(retain_loader, 'num_workers') else 0,
            pin_memory=retain_loader.pin_memory if hasattr(retain_loader, 'pin_memory') else False
        )
        return filtered_loader

    def unlearn(self, fa_threshold, ckpt_path):
        # KD teacher for the retain set. default: the model as it is when this
        # call starts. `kd_teacher` (unlearn.py: kd_teacher_path) overrides it,
        # used by sequential_unlearn.py so every stage distils from the ORIGINAL
        # base model instead of the already-unlearned checkpoint of the previous
        # stage (otherwise the damage of earlier stages is preserved by KD).
        teacher = getattr(self, 'kd_teacher', None)
        origin_model = teacher if teacher is not None else copy.deepcopy(self.model)
        print(f"[*] KD teacher: {'external (kd_teacher_path)' if teacher is not None else 'copy of the model at unlearn start'}")
        origin_model.eval()
        for param in origin_model.parameters():
            param.requires_grad = False

        total_unlearn_time = 0.0
        total_unlearn_steps = 0
        early_stop = False

        closest_fa_score = float('inf')

        if (self.selection_option != "diff"):
            print(f"[*] Starting unlearning with selection: {self.selection_option}, update_scope: {self.update_scope}")
        else:
            print(
                f"[*] Starting unlearning with selection: {self.selection_option}, alpha: {self.alpha} ,update_scope: {self.update_scope}")

        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()

            # expert selection (different options).
            selected_experts_per_layer = []
            if self.selection_option in ["diff", "ratio"]:
                forget_mass = self._get_routing_mass(self.forget_loader)
                retain_mass = self._get_routing_mass(self.retain_loader)
            elif self.selection_option == "gradient":
                grad_scores = self._get_gradient_influence(self.forget_loader)
            elif self.selection_option == "random":
                pass
            else:
                raise ValueError(f"Invalid selection option: {self.selection_option}")

            moe_layers = [m for _, m in self.model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']

            for l_idx, m in enumerate(moe_layers):
                if self.selection_option == "random":
                    generator = torch.Generator(device=self.device).manual_seed(epoch + (l_idx * 100))
                    selected_experts = torch.randperm(m.num_experts, generator=generator, device=self.device)[
                        :self.k_u].tolist()
                else:
                    if self.selection_option == "diff":
                        rho_m = forget_mass[l_idx] - self.alpha * retain_mass[l_idx]
                    elif self.selection_option == "ratio":
                        epsilon = 1e-8
                        rho_m = forget_mass[l_idx] / (retain_mass[l_idx] + epsilon)
                    elif self.selection_option == "gradient":
                        rho_m = grad_scores[l_idx]

                    if self.update_scope in ["all_experts", "full_model"]:
                        selected_experts = list(range(m.num_experts))
                    else:
                        _, topk_indices = rho_m.topk(self.k_u, dim=-1)
                        selected_experts = topk_indices.tolist()

                selected_experts_per_layer.append(selected_experts)
                m.allowed_experts = selected_experts
                m.unlearn_active_k = self.unlearn_active_k

            # apply update scope.
            self._apply_update_scope(selected_experts_per_layer, moe_layers)

            # re-init optimizer.
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            opt_class = type(self.optimizer)
            valid_kwargs = inspect.signature(opt_class.__init__).parameters.keys()
            filtered_defaults = {k: v for k, v in self.optimizer.defaults.items() if k in valid_kwargs}
            self.optimizer = opt_class(trainable_params, **filtered_defaults)

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"[*] Update Scope: {self.update_scope} | Trainable Params: {trainable_params_count:,} / {total_params:,} ({(trainable_params_count / total_params) * 100:.2f}%)"
                f" | unlearn_active_k: {self.unlearn_active_k if self.unlearn_active_k is not None else f'{self.k_u} (=k_u)'}")

            # filter retain set.
            epoch_retain_loader = self._create_filtered_retain_loader(self.retain_loader, selected_experts_per_layer,
                                                                      moe_layers)

            self.model.train()
            origin_model.eval()
            total_loss_accum = 0.0

            if epoch_retain_loader is not None:
                retain_iter = iter(epoch_retain_loader)
            else:
                retain_iter = None

            for step, forget_batch in enumerate(self.forget_loader):

                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)

                self.optimizer.zero_grad()

                # caculate loss forget & loss separation.
                logits_f, _ = self.model.forward_with_grad(images_f)
                loss_forget = self._unlearn_loss_forget(logits_f, labels_f)
                loss_sep = self._unlearn_loss_separation(moe_layers, selected_experts_per_layer)

                # only distill if retain loss is not empty.
                if retain_iter is not None:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(epoch_retain_loader)
                        retain_batch = next(retain_iter)

                    images_r = retain_batch[0].to(self.device)
                    labels_r = retain_batch[1].to(self.device)

                    logits_r, _ = self.model.forward_with_grad(images_r)
                    loss_retain = self._unlearn_loss_retain(logits_r, labels_r)
                    loss_distill = self._unlearn_loss_distill(logits_r, images_r, origin_model)
                else:
                    loss_retain = torch.tensor(0.0, device=self.device)
                    loss_distill = torch.tensor(0.0, device=self.device)

                total_loss = loss_forget + (self.beta * loss_retain) + (self.gamma * loss_distill) + (
                            self.eta * loss_sep)
                total_loss.backward()

                self.optimizer.step()

                total_loss_accum += total_loss.item()
                total_unlearn_steps += 1

            avg_loss = total_loss_accum / len(self.forget_loader)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time

            print(f"[*] End of epoch {epoch + 1}. Running full evaluation...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            self.model.train()

            if fa_threshold < fa_score < closest_fa_score:
                closest_fa_score = fa_score
                print(
                    f"[!] New closest FA found at epoch end: {closest_fa_score * 100:.2f}%. Saving fallback checkpoint...")
                torch.save(self.model.state_dict(), f"{ckpt_path}_closest_fa.pt")

            if fa_score <= fa_threshold:
                print(f"[*] Target condition met (FA = {fa_score * 100:.2f}% <= {fa_threshold * 100:.2f}%).")
                early_stop = True

            print(
                f"--> Epoch [{epoch + 1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | Steps: {total_unlearn_steps}")
            print(
                f"--> Metrics: RA: {ra_score * 100:.2f}% | FA: {fa_score * 100:.2f}% | TA: {ta_score * 100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)

            wandb.log({
                "epoch": epoch + 1,
                "unlearn_loss": avg_loss,
                "fa": fa_score,
                "ra": ra_score,
                "ta": ta_score,
                "mia": mia_score,
                "unlearn_steps_accum": total_unlearn_steps
            })

            # per-epoch router train/test consistency, same check learn() runs:
            # does each domain still route to the same top-k_u expert set on
            # held-out test images as on train images? during unlearning this
            # also shows whether the forget target's routing drifts while the
            # retained domains hold their assignment.
            if (
                    self.per_domain_train_loaders_eval
                    and self.per_domain_test_loaders
                    and (epoch + 1) % self.router_match_log_every == 0
            ):
                from metric.router_traintest_match import domain_train_test_expert_match
                match_metrics = domain_train_test_expert_match(
                    self.model,
                    self.per_domain_train_loaders_eval,
                    self.per_domain_test_loaders,
                    self.device,
                    k_u=self.router_match_k_u,
                    domain_names=self.domain_names,
                )
                if match_metrics:
                    match_metrics["epoch"] = epoch + 1
                    wandb.log(match_metrics)
                    print(f"  [router-match @ epoch {epoch + 1}] "
                          f"mass_exact={match_metrics['router_match/overall/mass_exact_match_rate']:.4f} "
                          f"sel_exact={match_metrics['router_match/overall/sel_exact_match_rate']:.4f} "
                          f"sel_tv={match_metrics['router_match/overall/sel_mean_tv']:.4f} "
                          f"(k_u={self.router_match_k_u}, "
                          f"gate_k={match_metrics['router_match/overall/gate_k']}, "
                          f"{match_metrics['router_match/overall/num_domains']} domains)")
                # `domain_train_test_expert_match` sets the model to eval();
                # restore train() before the next unlearning epoch.
                self.model.train()

            torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

            if early_stop:
                break

        print(f"[*] Unlearning finished. Total Steps: {total_unlearn_steps} | Total Time: {total_unlearn_time:.2f}s")

        self._run_final_router_diagnostics(phase="UNLEARN")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (
                    1024 ** 3) if torch.cuda.is_available() else 0.0

        wandb.log({
            "total_unlearn_time_sec": total_unlearn_time,
            "total_unlearn_steps": total_unlearn_steps,
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

        return total_unlearn_time