import json

for name, path in [("HONEST", "results/test_honest.json"), ("LEAKAGE", "results/test_leakage.json")]:
    d = json.load(open(path, encoding="utf-8"))
    t = d["training"]
    test = d["test_metrics"]
    bv = t["best_val_metrics"]
    print(f"\n{'='*60}")
    print(f"  MODE: {name}")
    print(f"{'='*60}")
    print(f"  best_epoch       = {t['best_epoch']}")
    print(f"  epochs_completed = {t['epochs_completed']}")
    print(f"  --- Monitor metrics (used for model selection) ---")
    print(f"  monitor AUC      = {bv['auc']:.4f}")
    print(f"  monitor AUPR     = {bv['aupr']:.4f}")
    print(f"  monitor samples  = {bv['num_samples']}")
    print(f"  --- Test metrics (final evaluation) ---")
    print(f"  test AUC         = {test['auc']:.4f}")
    print(f"  test AUPR        = {test['aupr']:.4f}")
    print(f"  test samples     = {test['num_samples']}")
    print(f"  --- Are monitor & test identical? ---")
    same = (bv['auc'] == test['auc']) and (bv['aupr'] == test['aupr'])
    print(f"  SAME = {same}")
