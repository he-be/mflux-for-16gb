# 計画: Krea 2 の DiT を M6 mini で 2〜3 倍速くする(実測ゲート方式)

- 日付: 2026-09-22
- ブランチ: `feat/krea2-block-streaming`
- 前提となる実測: [M8: 1 ブロックの時間の内訳](../../docs/16gb/measurements/2026-09-22-m8-block-profile.md)
- 管理先: fork [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)(upstream へ PR は出さない)

## 0. 前提(動かせない要件)

- 前の計画の要件はそのまま: **q8 固定**、**スワップは失敗**、実行は `tools/swapwatch.py` 越し、
  対象構成は q8 + 4step LoRA / 1024² と 1280² / 4 ステップ / euler / guidance 1.0 / seed 42。
- **画質を落とす手は使わない。** ここでやるのは「同じ計算を GPU が得意な形で出す」だけ。
  量子化ビット、LoRA、スケジューラ、解像度は触らない。
- **出力の数値が動く変更は画像で判定する**(M5b / M7 §5 と同じ)。同じ seed の画像を
  変更前と並べて見比べ、ピクセル差の統計も添える。「ビット一致」は要求しない
  (機材をまたいだ時点で既に一致しない。M7 §5)。
- M3 Pro 18GB でも壊さない(遅くならない、footprint が増えない)。

## 1. 何が遅いか(M8 の実測、M6 mini)

1 ステップ 21.2 s = 28 ブロック × (計算 610 + I/O 140) ms。計算 610 ms の内訳:

| | いま(float32) | bf16 なら | 差 |
|---|---|---|---|
| matmul 8 本 | 414 ms | 296 ms | 118 ms |
| attention | 142 ms | 28 ms | **114 ms** |
| その他 | 31 ms | ≈ 0 | 31 ms |
| **ブロック** | **587 ms** | **320 ms** | **267 ms** |

原因は `Krea2LatentCreator.create_noise` の float32 が 1 か所から DiT 全体に伝播していること。
チェックポイントは bf16 で、text encoder の出力も bf16。float32 なのはノイズだけ。

それとは別に 3 つ:

- **I/O が計算と直列**(`Krea2StreamedBlock.__call__`: 読む → 計算 → 捨てる の順)。
  mini の SSD は 3.34 GB/s で 1 ブロック 140 ms、ステップの 19%。
- **`mlp.down`(K=16384)の nax カーネルが半速**(8.9 TFLOPS)。K を 4 分割すると 15.7。
- **`mx.clear_cache()` をブロックごとに呼ぶ**のが +12〜15 ms / ブロック。

## 2. 段階(すべて M6 mini の実ウェイトで測る。M3 Pro は最後に回帰だけ)

順番は効果の大きい順ではなく、**画像判定が要るものを先に**片づける順。
M8a が通らなければ以降の見込みが全部変わるので、M8a を単独で確定させてから進む。

### M8a. 活性を bf16 にする ← 本命(見込み 21.2 → 13.1 s/step)

変更(`src/mflux/models/krea2/`):

- `Krea2Transformer.__call__` の入口で `hidden_states` を **bf16 に落とし**、出口で
  float32 に戻す。ノイズとサンプラ(Euler の `x + dt * v`)は float32 のまま。
  ノイズの dtype 自体は変えない(seed 42 のノイズ値を変えると比較が成り立たなくなる)。
  → `context`(bf16)との concat で昇格が起きなくなり、`tvec` も `img.dtype` 経由で bf16 になる。
- `Krea2Attention`: `mx.repeat` を消して sdpa に GQA のまま渡す(bf16 では速度差ゼロだが
  素直になる。float32 の経路が残った場合の保険にもなる)。
- RoPE / RMSNorm / QK norm の float32 往復は**そのまま**(合計 20 ms 弱で、精度のための設計)。
- 非ストリーミング(`mx.compile` あり)の経路でも同じ cast が効くことを確認。

測り方:

1. `tools/bench/qmm_spy.py` で 1 ステップの `quantized_matmul` を全部記録 →
   DiT の 224 本が **すべて bfloat16** になっていること(M7 §4 では float32 × 224)。
2. `MTL_CAPTURE_ENABLED=1 tools/bench/nax_probe.py` 相当で、本番の形が
   `affine_qmm_t_nax_bfloat16_*` に落ちること。
3. `--block-streaming` で 1024² と 1280² を swapwatch 越しに 1 回ずつ。
   ブロック計算 ≈ 320 / 560 ms、1 ステップ ≈ 13 / 20 s、footprint、swapouts、verdict。
4. **画像判定**: 変更前(M7 の `images/20260922-1414-m7-m6mini-1024.png` と `-1280.png`)と
   並べる。ピクセル差(違うピクセル数 / max / 平均)を M7 §5 の表と同じ形式で出す。
   同一機材で M7 の再現画像を撮り直してから比べる(M7 は同一機材で完全一致するので、
   差があれば全部 bf16 由来)。

合格基準:

- 4 本すべて `clean`、swapouts 0。footprint は M7 以下(活性が半分になるので下がるはず)。
- 1 ステップ **≤ 14 s**(1024²)、**≤ 21 s**(1280²)。
- 画像: 構図・被写体・光が同一で、描き込みの量が落ちていない。
  ピクセル差の平均が機材差(4.6 / 255)と同じ桁。

通らなかったときの代案(**画質側で落ちた場合だけ**):

- **残差ストリームだけ float32 に残す**: ブロック入口の `x` は float32 のまま、
  `prenorm(x)` / `postnorm(x)` の出力を bf16 にして attention / MLP に入れ、
  戻りを float32 に上げて足す。matmul と sdpa は bf16 で走り、残差の累積だけ float32。
  cast が 4 回増えて +8 ms / ブロック程度。M8 §3 の数字から見積もれる。
- それでも駄目なら「M6 mini の bf16 は不採用」と数字つきで結論し、M8b 以降だけやる。

### M8b. I/O を計算と重ねる(見込み 13.1 → 9.2 s/step)

`Krea2StreamedBlock` を「ブロック i の計算を GPU に投げたあと、i+1 を読む」に変える。

- 案 A(先に試す): `mx.async_eval(out)` で計算を投げてから、次のブロックの
  lazy ハンドルを `mx.stream(mx.cpu)` 上で `mx.async_eval` する。`mx.load` の実体化は
  CPU ストリームで動くので GPU の計算と並ぶはず。
- 案 B: Python スレッドで次のブロックを `mx.eval` する(MLX の eval は GIL を放す)。
- どちらも常駐が **+1 ブロック(461 MB)** になる。5.32 → 約 5.8 GB の見込みで、上限まで
  まだ 7 GB ある。

測り方: `Krea2BlockStream.record` の `io` が「見えている I/O」になるよう計り直し
(読み始めから bind までではなく、**計算の終わりから次の計算の始まりまで**の待ち)。
1024² / 1280² を swapwatch 越しに 1 回ずつ。

合格基準:

- 見えている I/O **≤ 10 ms / ブロック**(140 → 10)。
- 1 ステップ ≤ 10 s(1024²)。
- footprint +0.6 GB 以内、swapouts 0、`clean`。
- **出力は M8a と同一機材でピクセル完全一致**(カーネルも積算順も変わらないので、
  一致しなければバグ)。

### M8c. `mlp.down` を K で 4 分割する(見込み −40 ms / ブロック、−1.1 s/step)

- 読み込み時(`Krea2BlockStream.read` の直後、または bake 時)に `mlp.down` の
  `weight`(6144×4096 u32)/ `scales` / `biases`(6144×256)を K 方向に 4 等分し、
  4 本の `QuantizedLinear` の和として計算する。group_size 64 は 4096 を割り切るので
  scales の境界は崩れない。
- 非ストリーミングの経路にも入れるなら `Krea2SwiGLU` 側で持つ。
  まず `--block-streaming` だけで効果を測る。

測り方: `block_profile.py` にこの経路を足して 320 → 280 ms を確認、次に本番 1 回。
合格基準: 1 ステップ −1 s 以上、画像判定は M8a と同じ形式(積算順が変わる)。
**M8a の画像判定と同じ 1 回で済ませたいが、切り分けのため別々に測る。**

MLX 側の問題でもあるので、K=16384 で半速になる件は再現最小例を添えて
upstream の mlx に issue を出す価値がある(fork 内の対処とは独立)。

### M8d. `mx.clear_cache()` の回数を減らす(見込み −12 ms / ブロック)

- `clear_cache` をやめて `mx.set_cache_limit(1 GB 程度)` に置き換え、drop した
  ウェイトは上限超過分として解放され、活性のバッファは再利用される形にする。
- 合格基準: footprint が M8b と同じ(±0.2 GB)、`clean`、ブロック −10 ms 以上。
  footprint が増えるなら「4 ブロックに 1 回 clear」など回数で妥協するか、やめる。

### M8e. 持続負荷でクロックが落ちるかを測る(変更なし、判断材料だけ)

M8 §5 のとおり、冷えた状態で 17.0 TFLOPS、2 分の連続負荷のあとは 14.5 TFLOPS。
本番の 4 ステップ(bf16 化後は 40〜60 s)がどちらの領域かで、M8a〜d の到達点が
1 割変わる。M8a の本番実行中に `sudo powermetrics --samplers gpu_power -i 1000` で
GPU 周波数と電力を並走させ、ステップごとの `compute_ms` と並べる。
落ちているなら「mini の持続性能はこれ」と記録するだけで、対策はしない
(ファン制御はこの計画の外)。

### M8f. M3 Pro 18GB で回帰(最後に 1 回)

M8a〜d を入れた状態で M3 Pro の 1024² を 1 回。合格基準: 29.6 s/step より遅くならず、
footprint 5.21 GB より増えず、`clean`。M3 Pro は bf16 の qmm が 5.10 / float32 4.69 TFLOPS
なので、少し速くなるはず。sdpa の float32 → bf16 分も効く。

## 3. 到達点の見込み(1024²、M6 mini、1 ステップ)

| 段 | 計算 / ブロック | 見える I/O | 1 ステップ | 4 ステップ |
|---|---|---|---|---|
| M7(いま) | 610 ms | 140 ms | 21.2 s | 85 s |
| M8a | 320 ms | 140 ms | 13.1 s | 52 s |
| + M8b | 320 ms | 0 | 9.2 s | 37 s |
| + M8c + M8d | ≈ 266 ms | 0 | 7.6 s | 30 s |
| 理論下限 | 240 ms | 0 | 6.9 s | 28 s |

生成全体(TE 2 s + DiT + VAE 6 s)は 93〜101 s → **40 s 前後**。
1280² は 154 s → 70 s 前後。

## 4. 触るファイル

| 段 | ファイル |
|---|---|
| M8a | `model/krea2_transformer/transformer.py`(入口 / 出口の cast)、`attention.py`(repeat を消す) |
| M8b | `weights/krea2_weight_stream.py`(`Krea2StreamedBlock` / `Krea2BlockStream`) |
| M8c | `weights/krea2_weight_stream.py`(読み込み時の分割)または `feed_forward.py` |
| M8d | `weights/krea2_weight_stream.py` |
| 計器 | `tools/bench/block_profile.py`、`qmm_shapes.py`(済)、`qmm_spy.py`(dtype 列は既にある) |
| 記録 | `docs/16gb/measurements/2026-09-22-m8*.md`、`runs/m6mini/`、`images/` |

## 5. 測定の作法(この計画で足すもの)

- **GPU を独占する。** 別タスクが同じ GPU を使っていると数字が全部 5〜10% ずれて、
  しかも一貫してずれるので気づかない。測る前に `ps aux | grep python` を両方の機械で見る。
- **合成ウェイトは scales の dtype を合わせる。** `nn.quantize` を float32 のモジュールに
  掛けると float32 scales になり、bf16 の活性でも別のカーネルに落ちる(M8 §5)。
  ブロック単体の数字は `--model` で実チェックポイントを読んで取る。
- **冷えた状態と温まった状態を分ける。** 短いプローブは冷えた数字、本番は温まった数字。
  同じ形で 17.0 と 14.5 の両方が出るので、比べるときは条件を揃える。
- **画像判定は M7 の再現画像を同じ日に撮り直してから。** 同一機材なら完全一致するので、
  差が出れば全部変更由来と言える。

## 6. 非目標

- upstream への PR(mlx 側への issue は別)。
- text encoder / VAE の高速化(合わせて 8 s。DiT が 85 s)。
- RoPE / RMSNorm の float32 往復をやめること(合計 20 ms 弱で、精度側の設計)。
- q4 / q3、解像度を下げる、ステップを減らす。
- ファン制御や `powermetrics` の結果に対する対策。
