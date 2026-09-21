import os
import torch
import wandb

from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia

class Gradient_Ascent:
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
        device="cuda"
    ):
        self.model = model
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.unseen_loader = unseen_loader

        self.forget_loader = forget_loader
        self.forget_test_loader = forget_test_loader

        self.retain_loader = retain_loader
        self.retain_test_loader = retain_test_loader

        self.optimizer = optimizer
        self.criteria = criteria

        self.num_epoch = num_epoch
        self.device = device

        # non-member pool for the MIA metric. defaults to unseen_loader, but that
        # loader is augmented and (in class/domain settings) covers every class, so
        # entry points should override it via set_mia_unseen_loader().
        self.mia_unseen_loader = unseen_loader

        # evaluate() cost: FA(forget) + RA(retain-train) + TA(test) + MIA. RA and MIA
        # dominate -- on PACS class they forward ~8k images/epoch vs ~1.4k trained.
        # eval_every > 1 computes RA/TA/MIA only on scheduled epochs and reuses the
        # last values otherwise. FA is ALWAYS recomputed because fa_threshold early
        # stopping depends on it. A full evaluation is forced on the first and last
        # epoch, and whenever FA drops to eval_full_below, so an early-stopping run
        # never reports stale RA/TA/MIA. eval_every=1 is the original behaviour.
        self.eval_every = 1
        self.eval_full_below = 0.15
        self._eval_calls = 0
        self._last_full = None

    def set_mia_unseen_loader(self, loader):
        """
        held-out samples used as the non-member side of the MIA.

        must be matched to the forget set in class/domain composition and use the
        same deterministic transform, otherwise the attack separates the two sets
        on preprocessing or class identity instead of on membership. kept separate
        from `unseen_loader`, which some algorithms (e.g. SG_Unlearning) consume
        as augmented training data.
        """
        self.mia_unseen_loader = loader

    def learn(self, ckpt_path):
        self.model.train()
        
        for epoch in range(self.num_epoch):
            total_loss = 0.0
            
            for batch in self.train_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                loss = self.criteria(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(f"epoch [{epoch+1}/{self.num_epoch}] | loss: {avg_loss:.4f} | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")


    def _step_early_stop(self, fa_threshold, step, epoch, ckpt_path):
        """Kiểm tra FA sau mỗi `eval_every_steps` bước. Trả về True nếu cần dừng."""
        n = int(getattr(self, "eval_every_steps", 0) or 0)
        if n <= 0 or fa_threshold < 0 or step % n != 0:
            return False
        from metric.fa import forget_acc
        fa_score = forget_acc(self.model, self.forget_test_loader, self.device)
        self.model.train()
        print(f"    [step {step}] FA: {fa_score*100:.2f}%")
        wandb.log({"step": step, "fa_step": fa_score})
        if fa_score <= fa_threshold:
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            print(f"--> [step {step}, epoch {epoch+1}] early stop (FA <= {fa_threshold})")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            wandb.log({"epoch": epoch+1, "stop_step": step, "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score})
            torch.save(self.model.state_dict(), f"{ckpt_path}_stop_step{step}.pt")
            return True
        return False

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.train()
        step = 0
        stopped = False
        for epoch in range(self.num_epoch):
            total_loss = 0.0
            
            for batch in self.forget_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                loss = self.criteria(logits, labels)
                
                # reverses the loss sign to perform gradient ascent
                # the optimizer will try to minimize (-loss), which effectively maximizes the actual (loss)
                ascent_loss = -loss
                ascent_loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                step += 1
                if self._step_early_stop(fa_threshold, step, epoch, ckpt_path):
                    stopped = True
                    break
            if stopped:
                break
                
            avg_loss = total_loss / len(self.forget_loader)
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Loss: {avg_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)
            
            wandb.log({
                "epoch": epoch+1, 
                "unlearn_loss": avg_loss, 
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
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

    def evaluate(self):
        self._eval_calls += 1
        fa_score = forget_acc(self.model, self.forget_test_loader, self.device)

        due = (
            self.eval_every <= 1
            or self._last_full is None                      # first call
            or self._eval_calls % self.eval_every == 0
            or self._eval_calls >= self.num_epoch           # last epoch
            or fa_score <= self.eval_full_below             # about to early-stop
        )
        if due:
            ra_score = retain_acc(self.model, self.retain_test_loader, self.device)
            ta_score = test_acc(self.model, self.test_loader, self.device)
            # members = the forget set, non-members = held-out samples matched to it
            mia_score = mia(self.model, self.forget_test_loader, self.mia_unseen_loader, self.device)
            self._last_full = (ra_score, ta_score, mia_score)

        ra_score, ta_score, mia_score = self._last_full
        return fa_score, ra_score, ta_score, mia_score