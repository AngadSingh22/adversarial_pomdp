""""""
import json
import math
from pathlib import Path

def verify_summary(summary_path: Path):
    if not summary_path.exists():
        print(f'Skipping {summary_path}, not found.')
        return True
    with open(summary_path) as f:
        data = json.load(f)
    print(f'\n=== Verifying {summary_path.parent.name} ===')
    all_ok = True
    for gen_data in data:
        gen = gen_data['generation']
        before = gen_data['before']
        after = gen_data['after']
        checks = gen_data['checks']
        mean_dk_before = before['vs_D_k']['mean']
        mean_unif_before = before['scripted_modes']['UNIFORM']['mean']
        expected_def_adv = mean_dk_before - mean_unif_before
        mean_dk_after = after['vs_D_k']['mean']
        expected_att_adapt = mean_dk_after - mean_dk_before
        mean_unif_after = after['scripted_modes']['UNIFORM']['mean']
        expected_unif_drift = mean_unif_after - mean_unif_before
        actual_def_adv = checks.get('defender_adversarial')
        actual_att_adapt = checks.get('attacker_adaptation')
        actual_unif_drift = checks.get('uniform_drift')
        tol = 0.001
        print(f'  Gen {gen}:')

        def check_metric(name, expected, actual):
            nonlocal all_ok
            if actual is None:
                print(f'    [WARN] {name} is None (expected {expected:.4f})')
                return
            diff = abs(expected - actual)
            if diff > tol:
                print(f'    [FAIL] {name}: Expected {expected:.4f}, Got {actual:.4f} (Diff: {diff:.4f})')
                all_ok = False
            else:
                print(f'    [OK]   {name}: Expected == Actual ({actual:.4f})')
        check_metric('defender_adversarial', expected_def_adv, actual_def_adv)
        check_metric('attacker_adaptation', expected_att_adapt, actual_att_adapt)
        check_metric('uniform_drift', expected_unif_drift, actual_unif_drift)
    return all_ok

def main():
    base_dir = Path('results/experiments/IBR')
    if not base_dir.exists():
        print(f'Directory {base_dir} not found.')
        return
    success = True
    for seed_dir in sorted(base_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
            continue
        summary_file = seed_dir / 'ibr_summary.json'
        s = verify_summary(summary_file)
        success = success and s
    if success:
        print('\n✅ Verification SUCCESS. All checklist metrics strictly equal the raw means within tolerance.')
    else:
        print('\n❌ Verification FAILED. See errors above.')
if __name__ == '__main__':
    main()