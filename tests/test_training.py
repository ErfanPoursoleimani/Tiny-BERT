import math

import torch
from torch.utils.data import DataLoader

from mini_bert.builders import build_model
from mini_bert.config import MLMConfig
from mini_bert.data.collator import PretrainingCollator
from mini_bert.data.dataset import PretrainingDataset
from mini_bert.model import BertForPreTraining
from mini_bert.training.optimizer import build_optimizer
from mini_bert.training.scheduler import build_scheduler
from mini_bert.training.trainer import Trainer


def test_optimizer_param_groups(cfg):
    m = BertForPreTraining(cfg.model)
    opt = build_optimizer(m, 1e-3, 0.01)
    assert opt.param_groups[0]["weight_decay"] == 0.01 and opt.param_groups[1]["weight_decay"] == 0.0
    n = sum(len(g["params"]) for g in opt.param_groups)
    assert n == len({id(p) for p in m.parameters()})
    nd = {id(p) for p in opt.param_groups[1]["params"]}
    assert id(m.bert.embeddings.layer_norm.weight) in nd and id(m.bert.embeddings.layer_norm.bias) in nd


def test_scheduler_warmup_then_decay(cfg):
    m = BertForPreTraining(cfg.model)
    for kind in ("linear", "cosine"):
        opt = build_optimizer(m, 1e-3, 0.0); sch = build_scheduler(opt, 10, 100, 1e-3, 1e-4, kind)
        lrs = []
        for _ in range(100):
            lrs.append(sch.get_last_lr()[0]); opt.step(); sch.step()
        assert lrs[0] < lrs[9] and abs(lrs[9] - 1e-3) < 1e-9 and lrs[-1] < 2e-4


def test_gradients_finite(cfg):
    m = BertForPreTraining(cfg.model)
    ids = torch.randint(5, cfg.model.vocab_size, (2, 12)); labels = torch.full_like(ids, -100); labels[:, 4] = ids[:, 4]
    m(ids, labels=labels).loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n


def test_overfit_tiny_corpus(tok, tmp_path):
    """Whole pipeline sanity: tokenizer -> masking -> embeddings -> attention -> MLM head -> optimizer."""
    from mini_bert.config import Config, ModelConfig
    sents = ["The cat sat on the mat.", "The dog ran in the park.", "The bird flew over the house."]
    docs = [[tok.encode_ids(s) for s in sents]] * 1
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=128, max_position_embeddings=16, hidden_dropout_prob=0.0,
                            attention_probs_dropout_prob=0.0)
    cfg.data.max_seq_length = 16
    cfg.training.batch_size, cfg.training.gradient_accumulation_steps = 8, 1
    cfg.training.max_steps, cfg.training.warmup_steps = 150, 10
    cfg.training.learning_rate, cfg.training.min_learning_rate = 2e-3, 2e-4
    cfg.training.eval_every = cfg.training.save_every = 50; cfg.training.log_every = 10
    cfg.training.output_dir = str(tmp_path / "out"); cfg.training.checkpoint_dir = str(tmp_path / "ck")
    cfg.__post_init__()
    ds = PretrainingDataset(docs * 40, tok.cls_id, tok.sep_id, 16, seed=0, short_seq_prob=0.0)
    cf = lambda seed: PretrainingCollator(tok.pad_id, tok.mask_id, tok.vocab_size, tok.special_ids, cfg.mlm, seed=seed)  # noqa: E731
    tr = Trainer(cfg, build_model(cfg), ds, ds, cf, quiet=True)
    s = tr.train()
    vals = [h for h in tr.history if h["type"] == "val"]
    trains = [h for h in tr.history if h["type"] == "train"]
    print("initial loss", trains[0]["train_mlm_loss"], "final", trains[-1]["train_mlm_loss"],
          "val acc", vals[0]["mlm_acc"], "->", vals[-1]["mlm_acc"])
    assert trains[-1]["train_mlm_loss"] < 0.5 * trains[0]["train_mlm_loss"]
    assert vals[-1]["mlm_acc"] > vals[0]["mlm_acc"]
