# M3 Pro 18GB: 計算律速と SSD ストリーミングの実測

- 測定日: 2026-09-22
- 機材: MacBook Pro M3 Pro 18GB (GPU 18 コア), macOS 24.6.0, 内蔵 SSD
- ソフト: mflux `ada5323` (0.20.0 直後), MLX 0.32.0, Python 3.14
- 測定スクリプト: 本文中にそのまま再現できる形で記載

この文書の結論: **Krea 2 の推論は計算律速で、ウェイトを SSD からブロック単位で流し込んでも
I/O は計算時間に完全に隠れる。したがって RAM 容量はこの機材の制約ではない。**

## 1. GPU の行列積スループット

Krea 2 の DiT の代表形状 (hidden 6144、トークン 5120 = 1024² の画像 4096 + テキスト ~1024)。

```python
import mlx.core as mx, time
D, T = 6144, 5120
x = mx.random.normal((T, D)).astype(mx.bfloat16); mx.eval(x)
W = mx.random.normal((D, D)).astype(mx.bfloat16); mx.eval(W)
def bench(f, n=10):
    mx.eval(f()); mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())          # ループ内で eval しないと グラフ構築しか測れない
    mx.synchronize()
    return (time.perf_counter() - t) / n
```

| 形式 | 1 回 | 実効 |
|---|---|---|
| bf16 `matmul` | 65.5 ms | **5.90 TFLOPS** |
| q4 `quantized_matmul` (group 64) | 75.3 ms | **5.13 TFLOPS** |
| q8 `quantized_matmul` (group 64) | 75.7 ms | **5.11 TFLOPS** |

量子化行列積は bf16 の約 87%。ビット数による速度差はほぼ無い(4bit と 8bit が同じ)。
これは MLX が重みを展開してから float の行列積に渡す構造と整合する
(研究メモ §6 参照)。

## 2. 内蔵 SSD の読み出し帯域

ページキャッシュを避けるため `F_NOCACHE` (fcntl 48) を立てて 8.6GB を読む。

| | 帯域 |
|---|---|
| 書き込み (8.6GB, fsync 込み) | 5.34 GB/s |
| 読み出し 1 回目 (uncached) | 4.58 GB/s |
| 読み出し 2 回目 (uncached) | 5.24 GB/s |

## 3. ストリーミングの実証

28 個 × 138MB (合計 3.88GB) の safetensors を作り、**毎回 `mx.load` し直して**
matmul にかけ、参照を捨てる、を 3 周。Krea 2 の 28 ブロックを模したもの。

```python
for step in range(3):
    for i in range(28):
        w = mx.load(p)[f"b{i}.w"]     # lazy handle を毎回作り直す
        y = mx.matmul(x, w.T)
        mx.eval(y)
        del w, y
    mx.clear_cache()
```

| 周 | 累積時間 | peak RSS | MLX peak |
|---|---|---|---|
| 1 | 4.00 s | 0.17 GB | 0.32 GB |
| 2 | 7.59 s | 0.17 GB | 0.32 GB |
| 3 | 11.21 s | 0.17 GB | 0.32 GB |

- 1 周 3.6 s で 19.8 TFLOP を実行 = **5.5 TFLOPS**。§1 と一致し、計算律速のまま。
- 3.88GB を毎周流しても**常駐は 0.17GB**。MLX の lazy load は参照を捨てれば解放される。
- 周あたりの実効読み出しは約 1.08 GB/s。SSD の実力 (§2) の 1/5 で、隠れている。

## 4. Krea 2 への当てはめ

1024²、28 ブロック、13.1B パラメータ。1 ステップの計算量は
線形層 2×13.1e9×5120 ≈ 134 TFLOP + アテンション ≈ 18 TFLOP ≈ **150 TFLOP**。

| 形式 | 1 ステップの重み | SSD 時間 @5.0GB/s | 1 ステップの計算 @5.1TFLOPS | I/O 比率 |
|---|---|---|---|---|
| q4 | 7.4 GB | 1.5 s | 約 29 s | 5% |
| q8 | 14.0 GB | 2.8 s | 約 29 s | 10% |
| bf16 | 26.3 GB | 5.3 s | 約 26 s | 20% |

- bf16 でも隠れる。プリフェッチ (`mx.async_eval`) を入れれば比率はさらに下がる。
- 4 ステップなら denoise は約 2 分 (q8)。遅いが、スワップも OOM も起きない範囲。
- 常駐は「1〜2 ブロック + アクティベーション + 小物」で、1〜2GB 程度に収まる見込み。

## 5. 参考: 全ブロック常駐を前提にした場合(従来の見方)

`max_recommended_working_set_size` = **15.03 GB** (`memory_size` 19.33 GB)。

| 構成 | ロード時の常駐(重みのみ) | 判定 |
|---|---|---|
| mflux q8 (14.06 + TE 8.05 + VAE 0.51) | 22.6 GB | 不可 |
| mflux q4 (7.21 + 8.05 + 0.51) | 15.77 GB | 超過 |
| mflux q3 (5.61 + 8.05 + 0.51) | 14.17 GB | かろうじて |

この表は「全部 RAM に載せる」前提でのみ意味を持つ。§3 の結果により、
この前提を外すのが正しい方向で、q3 に落とす必要はない。

## 6. 未測定・注意

- Krea 2 実機での 1 ステップ時間 (§4 は FLOPS からの推定)。
- ブロック単位の bind/drop を 28 回/ステップ行うオーバーヘッド。
- MLX のアロケータとページキャッシュの相互作用 (`mx.clear_cache()` の頻度)。
- TE (Qwen3-VL-4B, bf16 8.05GB) をストリーミングした場合の挙動。エンコードは 1 回だけなので
  計算量が小さく、DiT ほど I/O が隠れない可能性がある。
- 長時間運転時の SSD の持続帯域と熱。
