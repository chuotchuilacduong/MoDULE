import os
import time
import torch
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class L1_Sparse(Gradient_Ascent):
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
        
        alpha=0.1,
        l1_mode="retain_ft",
        device="cuda"
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
        self.alpha = alpha
        if l1_mode not in ('retain_ft', 'ascent'):
            raise ValueError(f"l1_mode phải là 'retain_ft' hoặc 'ascent' (nhận {l1_mode!r})")
        self.l1_mode = l1_mode

    def unlearn(self, fa_threshold, ckpt_path):
        """l1-sparse unlearning (Jia et al., 2023, "Model sparsity can simplify machine
        unlearning"): fine-tune trên RETAIN với phạt l1 lên trọng số, hệ số gamma giảm
        tuyến tính về 0 qua các epoch.

            L = CE(retain) + gamma_t * sum |W|,   gamma_t = alpha * (1 - t / T)

        Bản trước của repo là  -CE(forget) + alpha * mean|W|  -- tức gradient ascent với một
        hằng số ~2e-3 gắn thêm (mean|W| ~ 0.02), ra số trùng GA đến 2 chữ số ở mọi dataset và
        không hề là l1-sparse của paper. `alpha` giờ là gamma_0 của phạt l1 dạng TỔNG
        (paper dùng 5e-4 cho ResNet-18; với DeiT-S 29M tham số nên bắt đầu ở ~5e-5).
        """
        self.model.train()
        total_unlearn_time = 0.0
        step = 0
        stopped = False
        l1_params = [p for n, p in self.model.named_parameters()
                     if p.requires_grad and 'weight' in n and 'bn' not in n and 'norm' not in n]

        ascent = (self.l1_mode == 'ascent')
        n_l1 = sum(p.numel() for p in l1_params)
        loader = self.forget_loader if ascent else self.retain_loader
        print(f"[*] l1_sparse mode={self.l1_mode}: "
              + ("-CE(forget) + alpha*mean|W| (công thức gốc của repo, ~GA)" if ascent
                 else "CE(retain) + gamma_t*sum|W| (Jia et al. 2023)"))
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_loss = 0.0
            gamma = self.alpha if ascent else self.alpha * (1.0 - epoch / max(self.num_epoch, 1))

            for batch in loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                ce_loss = self.criteria(logits, labels)
                l1_penalty = sum(p.abs().sum() for p in l1_params)
                if ascent:
                    batch_loss = -ce_loss + gamma * l1_penalty / max(n_l1, 1)
                else:
                    batch_loss = ce_loss + gamma * l1_penalty

                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss.item()
                step += 1
                if self._step_early_stop(fa_threshold, step, epoch, ckpt_path):
                    stopped = True
                    break
            if stopped:
                break

            avg_loss = total_loss / max(len(loader), 1)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time

            print(f"[*] evaluating epoch {epoch+1} (gamma={gamma:.2e})...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)

            wandb.log({
                "epoch": epoch+1,
                "unlearn_loss": avg_loss,
                "l1_gamma": gamma,
                "ra": ra_score,
                "fa": fa_score,
                "ta": ta_score,
                "mia": mia_score
            })

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_train_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time