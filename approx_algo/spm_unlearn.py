"""
SPM baseline for the MoDULE unlearning benchmark.

Ported/adapted from amberyzheng/spm_unlearning ("Designing to Forget: Deep
Semi-parametric Models for Unlearning", CVPR 2026). The paper's core claim is
that a semi-parametric model can be unlearned by *deleting support samples at
test time* -- no gradient steps, no retraining. This class implements exactly
that, wired into MoDULE's existing evaluate()/wandb/checkpoint conventions so
it drops into unlearn.py alongside gradient_ascent, scrub, salun, etc. and can
be compared apples-to-apples (same fa/ra/ta/mia metrics, same timing measurement).

Requires the model passed in to be an architecture.spm.SPMArchitecture instance
(it needs .build_support()); using this algo with any other architecture will
raise immediately.
"""
import time
import torch
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent
from architecture.spm import SPMArchitecture


class SPM_Unlearn(Gradient_Ascent):
    """
    unlearn() ignores optimizer/criteria entirely (no parameter update occurs).
    fa_threshold is accepted for interface parity but unused: since there is no
    iterative loop, there is nothing to early-stop.
    """

    def __init__(self, *args, support_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SPMArchitecture):
            raise TypeError(
                "SPM_Unlearn requires an SPMArchitecture model "
                f"(got {type(self.model).__name__}). Set model_name: spm_resnet18 "
                "(or spm_resnet34/spm_resnet50) in your config."
            )
        # Defaults to whatever the model was already built with.
        self.support_size = support_size or self.model.support_size

    @torch.no_grad()
    def _diagnose_support_bank(self):
        """FA=0.00 của SPM là quên thật hay là lớp bị xoá khỏi KHÔNG GIAN ĐẦU RA?

        PairwiseRelationExpert.forward khởi tạo `out = new_zeros(B, num_classes)` rồi
        CHỈ ghi vào các cột có mặt trong `unique(support_labels)`. Lớp nào vắng khỏi
        support bank thì cột của nó giữ nguyên 0 với MỌI input -- argmax không bao giờ
        chọn được nó, nên FA=0.00 là tất yếu về mặt cấu trúc, không phụ thuộc backbone
        còn mã hoá lớp đó hay không. Hàm này in ra bằng chứng.
        """
        lab = self.model.support_labels.detach().cpu()
        hist = torch.bincount(lab, minlength=self.model.num_classes)
        print(f"[SPM-DIAG] phân bố nhãn trong support bank (n={lab.numel()}): "
              + " ".join(f"c{c}:{hist[c].item()}" for c in range(self.model.num_classes)))
        missing = [c for c in range(self.model.num_classes) if hist[c].item() == 0]
        print(f"[SPM-DIAG] lớp VẮNG MẶT khỏi support bank: {missing}")

        self.model.eval()
        cols_zero = torch.ones(self.model.num_classes, dtype=torch.bool)
        n_seen, argmax_hist = 0, torch.zeros(self.model.num_classes, dtype=torch.long)
        for batch in self.forget_test_loader:
            logp, _ = self.model.inference(batch[0].to(self.device))
            probs = logp.exp()
            cols_zero &= (probs <= 1e-8).all(dim=0).cpu()
            argmax_hist += torch.bincount(probs.argmax(1).cpu(),
                                          minlength=self.model.num_classes)
            n_seen += probs.size(0)
            if n_seen >= 512:
                break
        dead = [c for c in range(self.model.num_classes) if cols_zero[c]]
        print(f"[SPM-DIAG] cột xác suất LUÔN = 0 trên {n_seen} ảnh forget: {dead}")
        print(f"[SPM-DIAG] dự đoán trên ảnh forget: "
              + " ".join(f"c{c}:{argmax_hist[c].item()}" for c in range(self.model.num_classes)))
        if missing and set(missing) == set(dead):
            print(f"[SPM-DIAG] => FA=0 do CẤU TRÚC: lớp {missing} bị xoá khỏi không gian "
                  f"đầu ra, không phải do backbone đã quên. MIA cũng suy biến vì "
                  f"log(clamp(0,1e-8)) = -18.420681 hằng số cho mọi mẫu.")

    def unlearn(self, fa_threshold, ckpt_path):
        start = time.time()

        # This is the entire unlearning operation: rebuild the memory bank
        # from retain_loader only, so forget-set samples are simply absent.
        n_support = self.model.build_support(self.retain_loader, max_support=self.support_size)

        total_unlearn_time = time.time() - start

        print(f"[*] SPM unlearning: rebuilt support bank from retain set "
              f"({n_support} samples) in {total_unlearn_time:.4f}s -- no gradient steps taken.")

        self._diagnose_support_bank()

        fa_score, ra_score, ta_score, mia_score = self.evaluate()
        print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | "
              f"TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")

        peak_memory_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        )
        wandb.log({
            "epoch": 1,
            "ra": ra_score,
            "fa": fa_score,
            "ta": ta_score,
            "mia": mia_score,
            "support_bank_size": n_support,
            "total_unlearn_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb,
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_1.pt")
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time
