# M8: DiT 1 ブロックの時間の内訳 → **遅さの正体は活性の float32**

- 測定日: 2026-09-22
- 機材: **Mac mini (Apple M6) 16GB**、macOS 27.0 (26A428)、MLX 0.32.0、GPU 12 コア(`applegpu_g18g`)
- 実装: `dcacf52`(`--block-streaming`)のブロックを、そのまま単体で回す
- 再現: `tools/bench/block_profile.py`(ブロック単位)、`tools/bench/qmm_shapes.py`(matmul の形ごと)
- 計画: [DiT を速くする](../../.cursor/plans/2026-09-22-krea2-dit-speed.md)

M7 の 1 ステップ 21 s は、M6 の GPU が出せる速さに対して遅すぎる。
matmul の総量は 1 ブロック **3.58 TFLOP**(1024²)で、nax の bf16 カーネル(17 TFLOPS)なら
210 ms で終わる計算なのに、実測は 610 ms。何がその差を食っているかを、
**実チェックポイントの block 0 を単体で回して**切り分けた。

## 0. 結論

| | float32 活性(いまの本番) | **bfloat16 活性** | |
|---|---|---|---|
| ブロック 1 回(1024²、4126 トークン) | 587 ms | **320 ms** | **1.84 倍** |
| ブロック 1 回(1280²、6430 トークン) | 1100 ms | **560 ms** | **1.96 倍** |
| うち matmul 8 本 | 414 ms(8.65 TFLOPS) | **296 ms**(12.1 TFLOPS) | |
| うち attention(sdpa) | 142 ms(2.95 TFLOPS) | **28 ms**(15.0 TFLOPS) | **5.1 倍** |
| うち残り(norm / 変調 / RoPE / 残差) | 31 ms | ≈ 0 | |
| ブロックのピークメモリ | 2.33 GB | **1.66 GB** | |

**DiT は最初から最後まで float32 で走っている。** 原因は 1 か所で、
`Krea2LatentCreator.create_noise` が `mx.random.normal` の既定 dtype(float32)で
ノイズを作り、それが `first` 層を経てブロックに入るから。チェックポイント側は
scales / biases / norm / 変調ベクトル / `first` / `last` まで**全部 bf16**
(safetensors ヘッダで確認)で、text encoder の出力も bf16。
float32 と bf16 を concat した時点で全体が float32 に昇格し、
以後 28 ブロックのすべての matmul が float32 の nax カーネル(10 TFLOPS)に落ち、
attention は float32 の sdpa(**3 TFLOPS**)に落ちる。

M7 §4 が「bf16 化で 610 → 360 ms」と見積もったのは matmul だけを見ていた。
**attention の float32 が同じくらい大きな穴**で、実測はそれより良い 320 ms。

## 1. ブロック単位の内訳(`block_profile.py --model krea2-lowram`)

実チェックポイント(`~/Library/Caches/mflux/16gb-bench/krea2-lowram`)の block 0、
ウォームアップ 3 回のあと 8 回平均(1280² は 5 回)。`matmuls` はブロックと同じ順に
8 本を鎖でつないだもの(attention だけ抜く)。`matmuls again` はブロックを測ったあと
もう一度同じものを測った数字で、**ズレていれば持続負荷でクロックが落ちている**印。

### 1024²(4126 トークン: 画像 4096 + テキスト 30)

```
float32 activations, block 0, tokens=4126   matmul 3.58 TFLOP  attention 0.42 TFLOP
  matmuls                                   414.3 ms    8.65 TFLOPS
  sdpa repeat                               142.0 ms    2.95 TFLOPS
  sdpa gqa                                  112.6 ms    3.71 TFLOPS
  block                                     587.3 ms    6.81 TFLOPS
  block+clear_cache                         598.6 ms    6.68 TFLOPS
  matmuls again                             412.8 ms    8.68 TFLOPS
  rest (block - matmuls - sdpa repeat)       31.1 ms
  peak memory 2.33 GB

bfloat16 activations, block 0, tokens=4126
  matmuls                                   296.2 ms   12.09 TFLOPS
  sdpa repeat                                27.8 ms   15.04 TFLOPS
  sdpa gqa                                   27.1 ms   15.42 TFLOPS
  block                                     319.5 ms   12.52 TFLOPS
  block+clear_cache                         334.4 ms   11.96 TFLOPS
  matmuls again                             300.4 ms   11.92 TFLOPS
  rest (block - matmuls - sdpa repeat)       -4.5 ms
  peak memory 1.66 GB
```

float32 の `block` 587 ms は M7 の本番実測 607〜610 ms と一致する
(差はストリーミング側の `block.update` と Python の分)。**計器は本番を再現している。**

### 1280²(6430 トークン)

```
float32:  matmuls 646.5 ms (8.63)   sdpa 353.0 ms (2.88)   block 1099.5 ms   +clear_cache 1124.2 ms   peak 3.00 GB
bfloat16: matmuls 496.7 ms (11.24)  sdpa  77.7 ms (13.07)  block  559.8 ms   +clear_cache  572.6 ms   peak 2.23 GB
```

M7 の本番 1118 ms とやはり一致。1280² では attention の比重が上がる
(1.02 TFLOP)ので、float32 の sdpa がブロックの **32%** を食っている。

### `mx.clear_cache()` の値段

ブロックごとに `clear_cache` を呼ぶと **+12〜15 ms / ブロック**(bf16 で 320 → 334 ms)。
解放したバッファを OS に返すので、次のブロックの活性が毎回ページフォルトから始まる。
28 ブロックで 0.4 s/ステップ。

### GQA の `mx.repeat`

k / v を 12 → 48 ヘッドに `mx.repeat` してから sdpa に渡している。sdpa は GQA を
そのまま受け取れるので、渡せば float32 で 142 → 113 ms、**bf16 では 27.8 → 27.1 ms
でほぼ差なし**。bf16 にすれば効かなくなる項目だが、コードは素直になるので一緒に直す。

## 2. matmul を形ごとに(`qmm_shapes.py --tokens 4126`)

冷えた状態で、形ごとに 1 本ずつ(ウォームアップ 3、10 回平均):

| 形(1 ブロックあたりの本数) | bf16 | float32 |
|---|---|---|
| wq / gate / wo `6144→6144`(×3) | 18.4 ms(**16.9** TFLOPS) | 31.1 ms(10.0) |
| wk / wv `6144→1536`(×2) | 4.8 ms(16.3) | 7.5 ms(10.4) |
| mlp gate / up `6144→16384`(×2) | 48.9 ms(**17.0**) | 80.3 ms(10.3) |
| **mlp down `16384→6144`**(×1) | **92.9 ms(8.9)** | 93.3 ms(8.9) |
| 　同じものを K で 2 分割して足す | 61.8 ms(13.4) | 90.6 ms(9.2) |
| 　同じものを **K で 4 分割**して足す | **52.8 ms(15.7)** | 89.5 ms(9.3) |
| 8 本の合計 | **255 ms** | 365 ms |

- **bf16 の nax カーネルは 17 TFLOPS 出る**(M7 の合成ベンチと一致)。float32 は 10。
- **`16384→6144` だけが dtype によらず 9 TFLOPS で止まる。** K が大きい形で
  nax カーネルが半分の速さになる。K を 4 つに割って足すと 15.7 TFLOPS まで戻る
  (**92.9 → 52.8 ms、1 ブロック −40 ms**)。積算順が変わるので出力の数値は動く。
- トークン数 4126(64 の倍数でない)は関係ない。4096 / 5120 でも同じ TFLOPS。
- 8 本を鎖でつないだ `matmuls`(296 ms)は個別合計(255 ms)より **40 ms 遅い**。
  この差の正体は未解明(§5)。

## 3. matmul でも attention でもない部分

一度きりの切り分け(bf16 活性、4126 トークン。スクリプトは残していない。
`block_profile.py` の `rest` 行で総量だけは再現できる):

| 演算 | 時間 |
|---|---|
| `Krea2RMSNorm`(float32 に上げて戻す) | 3.9 ms(bf16 のまま `rms_norm` なら 0.9) |
| `(1 + scale) * n + shift` | 1.7 ms |
| 残差 `x + gate * y` | 2.0 ms |
| `silu(gate) * up`(4126×16384) | 5.1 ms |
| `apply_rope`(float32、q 48 + k 12 ヘッド) | 8.4 ms |
| QK norm(q + k) | 4.8 ms |
| `mx.repeat` k, v | 1.2 ms |

全部足しても 30 ms 程度で、**bf16 のブロックでは matmul + sdpa の影に消える**
(`rest` ≈ 0)。RoPE / RMSNorm の float32 往復は精度のための設計で、値段は安い。
**ここは触らない。**

## 4. 見込み(1024²、M6 mini、1 ステップ)

M7 の実測は 28 × (610 計算 + 140 I/O + 0.4 drop) ms + グローバル層 ≈ 21.2 s。

| 手 | 計算 / ブロック | I/O(見える分) | 1 ステップ | 対 M7 |
|---|---|---|---|---|
| M7 のまま | 610 ms | 140 ms | 21.2 s | 1.0 |
| 活性を bf16 | **320 ms** | 140 ms | **13.1 s** | 1.6 倍 |
| + I/O を計算と重ねる | 320 ms | 0 | **9.2 s** | 2.3 倍 |
| + mlp down の K 分割、+ clear_cache を減らす | ≈ 266 ms | 0 | **7.6 s** | 2.8 倍 |
| 理論下限(17 TFLOPS matmul + 15 TFLOPS sdpa) | 240 ms | 0 | 6.9 s | 3.1 倍 |

1280² は 35.2 s → bf16 で 19.5 s → I/O を隠して **15.9 s**(2.2 倍)。

**この 3 つはどれも画質と無関係な工学の話**(q8 のまま、LoRA もそのまま)。
ただし bf16 化と K 分割は積算順が変わるので出力の数値は動く。
機材をまたいだときと同程度の差(M7 §5: 平均 4.6 / 255)に収まるはずだが、
**画像を見比べて判定する**(M5b と同じ扱い)。

## 5. 踏んだもの・未解明

- **GPU を独占して測る。** 最初の 1 周は mini で別タスクが走っていて、
  数字が全部 5〜10% ずれた。止めてから取り直した(この文書の数字は取り直し後)。
- **合成ブロックの scales を float32 にすると別物になる。** `nn.quantize` を
  float32 のモジュールに掛けると scales / biases が float32 になり、bf16 の活性を
  渡しても matmul が 458 ms(7.8 TFLOPS)まで落ちる(実ウェイトの bf16 scales なら
  296 ms)。**実チェックポイントは bf16 scales** なので本番には無関係だが、
  LoRA の焼き込みや将来の再量子化で scales が float32 に化けたら同じ穴に落ちる。
  `block_profile.py` は既定で bf16 scales、`--scales float32` で再現できる。
- **持続負荷でクロックが落ちる兆候。** `qmm_shapes.py` は冷えた状態で 17.0 TFLOPS、
  2 分の連続負荷のあとに回すと 14.5 TFLOPS(同じ形)。ブロック内の `matmuls again`
  では 1〜2% しかずれないので、数十秒では効かず、数分で効く。
  4 ステップの本番(90 s)がどちらの領域かは `powermetrics`(要 sudo)で
  GPU 周波数を見ないと分からない。**未測定。**
- **鎖でつないだ 8 本が個別合計より 40 ms 遅い**(296 vs 255 ms)。中間の活性
  (4126×16384 bf16 = 135 MB)の確保やカーネル間の依存待ちが疑わしいが未確認。
  理論下限との差のうち、これが最大の未解明分。
