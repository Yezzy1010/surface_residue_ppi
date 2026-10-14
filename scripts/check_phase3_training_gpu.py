"""Validate Phase 3 readout/export changes on existing real train/val caches."""
import json
import os
from pathlib import Path
import tempfile

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_phase2_task1 import Readout, load_examples, fit_statistics, make_batch, evaluate
from train_phase3_task1 import export_predictions


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path('.').resolve()
    protocol = root / 'runs/phase2_protocol/20260925/local_20250925_phase2_protocol_01'
    rows = json.loads((protocol / 'pilot_manifest.json').read_text())
    chosen = [x for split in ('train', 'val') for x in sorted(
        [r for r in rows if r['split'] == split], key=lambda r:r['lmdb_index'])[:8]]
    manifests = {s:{(r['split'], r['lmdb_index']):r for r in chosen if r['split'] == s}
                 for s in ('train', 'val')}
    keys = set(manifests['train']) | set(manifests['val'])
    features = json.loads((root / 'runs/phase2_features/275876_final_v2/report.json').read_text())
    features['records'] = [r for r in features['records'] if (r['split'], r['lmdb_index']) in keys]
    features['records_requested'] = len(chosen)
    examples, dataset = load_examples(root, features, chosen, manifests['train'], manifests['val'], manifests['val'])
    stats = fit_statistics(examples['train'])
    device = torch.device('cuda')
    checks = []
    out = root / 'runs/phase3_smoke' / os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    for variant in ('ES', 'ES-B', 'ES-C'):
        torch.manual_seed(20260927)
        model = Readout(variant).to(device)
        batch = make_batch(examples['train'][:4], stats, device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(batch)[batch['mask']], batch['labels'][batch['mask']])
        loss.backward()
        assert model.project[0].weight.grad is not None
        assert model.head[0].weight.grad is not None
        if model.adapter is not None:
            assert model.adapter.value.weight.grad is not None
        assert all(not batch[key].requires_grad for key in ('esm', 'gearnet', 'surface'))
        before = evaluate(model, examples['val'], stats, 4, device)
        after, files = export_predictions(model, examples['val'], stats, 4, device, out, variant)
        import numpy as np
        differences = {key: abs(before[key] - after[key]) for key in before}
        print('METRIC_ROUNDTRIP', variant, json.dumps(dict(before=before, after=after,
              absolute_differences=differences)), flush=True)
        # Compare scores, rather than rank metrics, across separate GPU passes:
        # scatter reductions may reorder near-tied scores. Independently verify
        # all reported metrics against sklearn on the same saved scores below.
        repeat = []
        with torch.no_grad():
            for start in range(0, len(examples['val']), 4):
                batch = make_batch(examples['val'][start:start+4], stats, device)
                repeat.append(model(batch)[batch['mask']].cpu().numpy())
        with np.load(files['path']) as predictions:
            print('SCORE_REPEAT_MAX_ERROR', variant,
                  float(np.max(np.abs(np.concatenate(repeat)-predictions['logits']))), flush=True)
            np.testing.assert_allclose(np.concatenate(repeat), predictions['logits'], rtol=1e-5, atol=1e-6)
            values = [average_precision_score(predictions['labels'][a:b], predictions['logits'][a:b])
                      for a,b in zip(predictions['offsets'][:-1], predictions['offsets'][1:])
                      if predictions['labels'][a:b].sum()]
            aucs = [roc_auc_score(predictions['labels'][a:b], predictions['logits'][a:b])
                    for a,b in zip(predictions['offsets'][:-1], predictions['offsets'][1:])
                    if 0 < predictions['labels'][a:b].sum() < b-a]
            assert abs(average_precision_score(predictions['labels'], predictions['logits'])
                       - after['pooled_average_precision']) < 1e-12
            assert abs(roc_auc_score(predictions['labels'], predictions['logits'])
                       - after['pooled_auroc']) < 1e-12
        assert abs(float(np.mean(values)) - after['average_precision']) < 1e-12
        assert abs(float(np.mean(aucs)) - after['auroc']) < 1e-12
        checks.append(dict(variant=variant, status='passed', metrics=after,
                           roundtrip_absolute_differences=differences))
    report = dict(status='passed', dataset=dataset, checks=checks, test_records_read=0, test_labels_read=0,
                  purpose='real-cache gradient/export validation, not method performance')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT', out / 'report.json', 'passed', flush=True)


if __name__ == '__main__':
    main()
