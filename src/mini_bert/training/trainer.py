"""Training loop: AMP, gradient accumulation, checkpointing, evaluation, logging."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..evaluation.mlm import autocast_ctx, evaluate
from ..model.bert import BertForPreTraining
from ..utils.device import PrecisionInfo, select_device_and_precision, print_hardware_banner
from ..utils.logging import get_logger, get_tb_writer
from ..utils.parameter_count import count_params
from ..utils.profiling import cpu_ram_mb, gpu_memory_mb, reset_peak
from ..utils.seed import set_seed
from .checkpoint import copy_checkpoint, load_checkpoint, save_checkpoint
from .optimizer import build_optimizer
from .scheduler import build_scheduler


class Trainer:
    def __init__(self, cfg: Config, model: BertForPreTraining, train_ds, val_ds, collator_factory,
                 tokenizer_path: str | None = None, run_dir: str | None = None,
                 precision: PrecisionInfo | None = None, quiet: bool = False) -> None:
        self.cfg, self.tc = cfg, cfg.training
        set_seed(self.tc.seed)
        self.prec = precision or select_device_and_precision(self.tc.precision)
        self.device = self.prec.device
        self.model = model.to(self.device)
        self.model.bert.set_gradient_checkpointing(self.tc.gradient_checkpointing)
        self.tokenizer_path = tokenizer_path
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir or os.path.join(self.tc.output_dir, "runs", f"{stamp}_{self.tc.experiment_name}")
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "samples"), exist_ok=True)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints") if run_dir else self.tc.checkpoint_dir
        self.log = get_logger(f"mini_bert.{id(self)}", os.path.join(self.run_dir, "training.log"))
        if quiet:
            self.log.setLevel("WARNING")
        self.tb = get_tb_writer(os.path.join(self.run_dir, "tensorboard"))
        g = torch.Generator().manual_seed(self.tc.seed)
        self.train_loader = DataLoader(train_ds, batch_size=self.tc.batch_size, shuffle=True, drop_last=True,
                                       collate_fn=collator_factory(None), generator=g,
                                       num_workers=self.tc.num_workers)
        # fixed seed for validation masks -> comparable across evaluations
        self.val_loader = DataLoader(val_ds, batch_size=self.tc.batch_size, shuffle=False,
                                     collate_fn=collator_factory(1234))
        self.optimizer = build_optimizer(self.model, self.tc.learning_rate, self.tc.weight_decay,
                                         fused=self.device.type == "cuda")
        self.scheduler = build_scheduler(self.optimizer, self.tc.warmup_steps, self.tc.max_steps,
                                         self.tc.learning_rate, self.tc.min_learning_rate, self.tc.scheduler)
        self.scaler = torch.amp.GradScaler("cuda") if self.prec.use_scaler else None
        self.step = 0
        self.epoch = 0
        self.best_val = float("inf")
        self.history: list[dict] = []

    # ------------------------------------------------------------------ utils
    def banner(self) -> None:
        tc, mc = self.tc, self.cfg.model
        eff = tc.batch_size * tc.gradient_accumulation_steps
        tokens_upd = eff * self.cfg.data.max_seq_length
        print_hardware_banner(self.prec)
        print(f"Model: params={count_params(self.model):,} layers={mc.num_hidden_layers} hidden={mc.hidden_size} "
              f"heads={mc.num_attention_heads} context={mc.max_position_embeddings}")
        print(f"Micro batch: {tc.batch_size}\nAccumulation: {tc.gradient_accumulation_steps}\n"
              f"Effective batch: {eff}\nTokens/update (max): {tokens_upd:,}\n" + "=" * 44)

    def _batches(self):
        while True:
            for b in self.train_loader:
                yield b
            self.epoch += 1

    def _save(self, name: str, final_metrics: dict | None = None) -> str:
        path = os.path.join(self.ckpt_dir, name)
        save_checkpoint(path, self.model, self.optimizer, self.scheduler, self.scaler, self.step, self.epoch,
                        self.cfg.to_dict(), self.tc.seed, self.best_val, self.tokenizer_path)
        return path

    def resume(self, path: str) -> None:
        st = load_checkpoint(path, self.model, self.optimizer, self.scheduler, self.scaler, map_location=self.device)
        self.step, self.epoch, self.best_val = st["step"], st["epoch"], st["best_val"]
        self.log.info(f"resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------ main loop
    def train(self, resume: str | None = None, max_steps: int | None = None) -> dict:
        tc = self.tc
        total = max_steps or tc.max_steps
        if resume:
            self.resume(resume)
        self.banner()
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            import yaml
            yaml.safe_dump(self.cfg.to_dict(), f)
        reset_peak()
        self.model.train()
        it = self._batches()
        win_tokens, win_samples, win_loss, win_n, t0 = 0, 0, 0.0, 0, time.time()
        win_mlm, win_nsp = 0.0, 0.0
        while self.step < total:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(tc.gradient_accumulation_steps):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in next(it).items()}
                with autocast_ctx(self.device, self.prec.dtype):
                    out = self.model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"],
                                     batch["labels"], batch.get("next_sentence_label"))
                loss = out.loss / tc.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {self.step}: {out.loss.item()}")
                (self.scaler.scale(loss) if self.scaler else loss).backward()
                win_loss += out.loss.item(); win_n += 1
                win_mlm += out.mlm_loss.item()
                win_nsp += out.nsp_loss.item() if out.nsp_loss is not None else 0.0
                win_tokens += int(batch["attention_mask"].sum()); win_samples += batch["input_ids"].size(0)
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip))
            if self.scaler:
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.step % tc.log_every == 0 or self.step == total:
                dt = max(time.time() - t0, 1e-9)
                mem = gpu_memory_mb()
                rec = {"step": self.step, "epoch": self.epoch, "train_loss": win_loss / win_n,
                       "train_mlm_loss": win_mlm / win_n, "lr": self.scheduler.get_last_lr()[0],
                       "grad_norm": gnorm, "tokens_per_sec": win_tokens / dt, "samples_per_sec": win_samples / dt,
                       "gpu_allocated_mb": mem["allocated"], "gpu_reserved_mb": mem["reserved"],
                       "gpu_peak_mb": mem["peak_allocated"], "cpu_ram_mb": cpu_ram_mb()}
                if self.cfg.nsp.enabled:
                    rec["train_nsp_loss"] = win_nsp / win_n
                self.history.append({"type": "train", **rec})
                self.log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
                if self.tb:
                    self.tb.add_scalar("Loss/train", rec["train_loss"], self.step)
                    self.tb.add_scalar("LearningRate", rec["lr"], self.step)
                    self.tb.add_scalar("GradientNorm", gnorm, self.step)
                    self.tb.add_scalar("TokensPerSecond", rec["tokens_per_sec"], self.step)
                    self.tb.add_scalar("GPU/AllocatedMemory", mem["allocated"], self.step)
                    self.tb.add_scalar("GPU/ReservedMemory", mem["reserved"], self.step)
                    if self.cfg.nsp.enabled:
                        self.tb.add_scalar("NSP/loss_train", rec["train_nsp_loss"], self.step)
                win_tokens = win_samples = win_n = 0; win_loss = win_mlm = win_nsp = 0.0; t0 = time.time()

            if self.step % tc.eval_every == 0 or self.step == total:
                self._eval_and_log()
            if self.step % tc.save_every == 0 or self.step == total:
                self._save("latest.pt")
                if self.step % tc.save_every == 0:
                    self._save(f"step_{self.step}.pt")
        summary = self.finish()
        return summary

    def _eval_and_log(self) -> dict:
        m = evaluate(self.model, self.val_loader, self.device, self.prec.dtype, self.tc.eval_batches)
        self.history.append({"type": "val", "step": self.step, **{k: v for k, v in m.items()}})
        self.log.info(f"[val] step={self.step} " + " ".join(
            f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)))
        if self.tb:
            self.tb.add_scalar("Loss/validation", m["mlm_loss"], self.step)
            self.tb.add_scalar("MLM/accuracy", m["mlm_acc"], self.step)
            self.tb.add_scalar("MLM/top5_accuracy", m["mlm_top5_acc"], self.step)
            if "nsp_acc" in m:
                self.tb.add_scalar("NSP/loss", m["nsp_loss"], self.step)
                self.tb.add_scalar("NSP/accuracy", m["nsp_acc"], self.step)
        if m["mlm_loss"] < self.best_val:
            self.best_val = m["mlm_loss"]
            self._save("best.pt")
        return m

    def finish(self) -> dict:
        trains = [h for h in self.history if h["type"] == "train"]
        vals = [h for h in self.history if h["type"] == "val"]
        mem = gpu_memory_mb()
        summary = {
            "steps": self.step, "final_train_loss": trains[-1]["train_loss"] if trains else None,
            "final_val": vals[-1] if vals else None, "best_val_mlm_loss": self.best_val,
            "peak_vram_mb": mem["peak_allocated"], "precision": self.prec.name, "device": str(self.device),
            "mean_tokens_per_sec": sum(t["tokens_per_sec"] for t in trains) / max(len(trains), 1),
            "params": count_params(self.model), "torch": torch.__version__, "seed": self.tc.seed,
        }
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump({"summary": summary, "history": self.history}, f, indent=1, default=str)
        try:
            from .plots import save_training_plots
            save_training_plots(self.history, os.path.join(self.run_dir, "plots"))
        except Exception as e:  # plotting must never kill a finished run
            self.log.warning(f"plotting failed: {e}")
        if self.tb:
            self.tb.close()
        return summary
