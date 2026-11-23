"""Compare every nonzero index and value in Dart with the fitted Python pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dart", default="dart")
    args = parser.parse_args()
    rows = pd.read_csv(ROOT / "data/arabic_expenses.csv").description.tolist()
    rows.extend(["دفعت فاتورة الإنترنت", "فَاتـُورَة ١٢٣،", "الا\u0655نترنت", "", "!!!",
                 "١٢٣", "ب ب ب", "تاكسي تاكسي", "دفعت WiFi", "🙂\tموقف/سيارة"])
    model = joblib.load(ROOT / "models/arabic.joblib")
    expected = model.calibrated_classifiers_[0].estimator[0].transform(rows).tocsr()
    result = subprocess.run([args.dart, "run", "tool/check_vectors.dart"], cwd=ROOT / "mobile",
                            input="".join(json.dumps(r) + "\n" for r in rows),
                            capture_output=True, encoding="utf-8")
    if result.returncode:
        sys.exit(result.stderr)
    vectors = result.stdout.splitlines()
    assert len(vectors) == len(rows)
    worst = 0.0
    for i, line in enumerate(vectors):
        actual = json.loads(line)
        ref = expected[i]
        assert set(map(int, actual)) == set(ref.indices), rows[i]
        values = [actual[str(j)] for j in ref.indices]
        np.testing.assert_allclose(values, ref.data, rtol=0, atol=1e-6, err_msg=rows[i])
        worst = max(worst, np.max(np.abs(values-ref.data), initial=0))
    print(f"{len(rows)} matching vectors; maximum absolute difference {worst:.9g}")
