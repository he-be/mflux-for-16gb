# Runs the real mflux krea2 CLI, recording every distinct quantized_matmul the
# model issues. The signatures are replayed separately under a Metal capture to
# see which kernel Metal actually dispatched.
import json
import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx

seen = Counter()
real_qmm = mx.quantized_matmul


def spy(x, w, scales=None, biases=None, transpose=True, group_size=64, bits=4, **kw):
    seen[(tuple(x.shape), tuple(w.shape), transpose, group_size, bits, str(x.dtype))] += 1
    return real_qmm(x, w, scales=scales, biases=biases, transpose=transpose, group_size=group_size, bits=bits, **kw)


mx.quantized_matmul = spy

sys.argv = ["mflux-generate-krea2"] + sys.argv[1:]
from mflux.models.krea2.cli.krea2_generate import main  # noqa: E402

try:
    main()
finally:
    mx.quantized_matmul = real_qmm
    rows = [
        {"x": list(k[0]), "w": list(k[1]), "transpose": k[2], "gs": k[3], "bits": k[4], "dtype": k[5], "calls": v}
        for k, v in seen.most_common()
    ]
    Path("/tmp/qmm_signatures.json").write_text(json.dumps(rows, indent=1))
    print(f"\n=== {len(rows)} distinct quantized_matmul signatures, {sum(seen.values())} calls ===")
    for r in rows:
        print(f"  x={r['x']} w={r['w']} gs={r['gs']} bits={r['bits']} {r['dtype']} x{r['calls']}")
