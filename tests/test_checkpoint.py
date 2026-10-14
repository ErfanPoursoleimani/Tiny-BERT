import torch

from mini_bert.model import BertForPreTraining
from mini_bert.training.checkpoint import load_checkpoint, save_checkpoint
from mini_bert.training.optimizer import build_optimizer
from mini_bert.training.scheduler import build_scheduler


def test_roundtrip_equivalent_outputs(cfg, tmp_path):
    m = BertForPreTraining(cfg.model)
    opt = build_optimizer(m, 1e-3, 0.01); sch = build_scheduler(opt, 2, 10, 1e-3, 1e-4)
    ids = torch.randint(5, cfg.model.vocab_size, (2, 8)); labels = ids.clone()
    m(ids, labels=labels).loss.backward(); opt.step(); sch.step()
    p = str(tmp_path / "c.pt")
    save_checkpoint(p, m, opt, sch, None, step=1, epoch=0, config=cfg.to_dict(), seed=1, best_val=1.0)
    m2 = BertForPreTraining(cfg.model)
    opt2 = build_optimizer(m2, 1e-3, 0.01); sch2 = build_scheduler(opt2, 2, 10, 1e-3, 1e-4)
    st = load_checkpoint(p, m2, opt2, sch2)
    assert st["step"] == 1 and st["best_val"] == 1.0
    m.eval(); m2.eval()
    assert torch.allclose(m(ids).mlm_logits, m2(ids).mlm_logits, atol=1e-6)
    assert sch2.last_epoch == sch.last_epoch
    assert m2.mlm_head.decoder.weight is m2.bert.embeddings.token_embeddings.weight  # tie preserved
