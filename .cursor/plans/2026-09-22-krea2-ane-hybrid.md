# 計画: Krea 2 の DiT に ANE を 2 台目の演算器として足す — probe と MLP の列分割

- 日付: 2026-09-22
- ブランチ: `feat/krea2-block-streaming`
- 前提となる実測: [M9: 天井 19 TFLOPS](../../docs/16gb/measurements/2026-09-22-m9-ceiling-probe.md)、
  [M8 §2: 形ごとの matmul 時間](../../docs/16gb/measurements/2026-09-22-m8-block-profile.md)、
  [M9c: 7.90 s/step の現状](../../docs/16gb/measurements/2026-09-22-m9c-prefetch-interference.md)
- 前の計画: [DiT を天井に近づける](2026-09-22-krea2-dit-ceiling.md)(M9a〜e 済。GPU 単独の伸びしろは 1.27 倍まで)
- 外部の根拠: `~/dev/Irodori-TTS`(`docs/note-mac.md`、`docs/experiments/17-m1-ane-factors.md`、`irodori_tts/ane_worker.py`)、
  Draw Things の ANE 記事(§7)
- 管理先: fork [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)(upstream へ PR は出さない)
- 状態: **M11a 着手中。`tools/bench/ane_probe.py` は書けたが mini では未実行。** 再開手順は §8。

## 0. 前提(動かせない要件)

前の計画と同じ。**q8 固定**、**スワップは失敗**、`tools/swapwatch.py` 越し、対象構成は
q8 + 4step LoRA / 1024² と 1280² / 4 ステップ / euler / guidance 1.0 / seed 42。
**出力の数値が動く変更は画像で判定する**(同じ seed を並べ、ピクセル差の統計を添える。M8a の形式)。
追加で 2 つ:

- **ANE 側のメモリも footprint に数える。** Core ML は別プロセスになる(§2)ので、`swapwatch.py` が
  ツリー全体を合計していることを頼りにし、`mx.get_peak_memory()` だけで判定しない。
- **常駐モード(`--block-streaming` なし)は触らない。** 既存テストのビット一致を保つ。

## 1. 問い

**「M6 は GPU に neural accelerator、それとは別にデュアル ANE を持つ。演算器を足せば速くなるのではないか。」**

先に出した「ANE は割に合わない」という答えは **ANE で GPU を置き換える**前提で、それは撤回する。
根拠は 3 つ。

1. **Irodori-TTS(M3 Pro、`docs/note-mac.md`)。** DiT のステップを ANE に、CFG の 3 分岐のうち 1 本を GPU に
   投げて同時に回し **1.35 倍**。「ANE 単体が速いというより、演算器が 2 つになることのほうが効いている。
   ANE だけに全部載せた版はもっと遅かった。」同時に踏んだこと: **メモリは 1 バイトも減らない**、
   Core ML の `predict` が GIL を握るので **別プロセス**、入力の形ごとに別モデル(23 種を同梱)、
   半精度で数値が壊れる箇所があった、事前コンパイル 30 分 / ディスク 8.4 GB。
2. **Draw Things(§7)。** 独自ランタイムのまま **行列積だけ**を Core ML にコンパイルして ANE で回す。
   int8 重み + 行ごとのスケール(8-bit S)、fp16 の活性。M4 で最大 1.8 倍、ANE の実効は
   **約 22 TOPS**(公称 38)。**M5 では ANE+NAX の hybrid を既定にしている。** ANE 単独の経路は
   「省電力の選択肢で、生成時間は GPU より少し長い」。
3. **M6 のデュアル ANE(§7)。** Apple は「ピーク 2 倍、システムフレームワークが 2 基を自動で使う」と言っている。
   実効の数字は公開されていない。

## 2. なぜ TTS と同じ形にはできないか → 行列積を列で割る

TTS には CFG の分岐という自然な並列があった。Krea 2 Turbo は guidance 1.0 で分岐がなく、
ブロックの中も直列(`transformer_block.py:21-22`):

```python
x = x + pregate  * self.attn(...)   # 先
x = x + postgate * self.mlp(...)    # attn の結果に依存
```

だから「別の分岐を ANE に投げる」ではなく、**MLP の行列積を出力列で GPU と ANE に分けて同時に走らせる**。
`Krea2SwiGLU`(`feed_forward.py`)は列で閉じる:

- `gate` / `up`(6144→16384)の出力列を GPU:ANE = (1−a):a に分ける
- `silu(gate) * up` は列ごとの要素演算なので各自で完結
- `down`(16384→6144)は各自が持つ列 = K の一部で部分積を出し、**最後に足す**(M9c の K4 分割と同じ積算。
  順序が変わるので数値は動く → 画像判定)

同期はブロックあたり 1 回。ANE へ渡すのは MLP の入力 `(1+postscale)*postnorm(x)+postshift`
(4126×6144 fp16 = 50.7 MB)、戻るのは `down` の部分積(同 50.7 MB)。中間 4126×16384 は ANE の中に留まる。
attention は GPU のまま(sdpa は既に天井、Draw Things も行列積しか ANE に出していない)。
ANE の取り分の重みは **常駐**(ANE はブロック単位のストリーミングに向かない)。その分 GPU 側の
ストリーミングは軽くなる(1 ブロック 461 MB → 461 − a×320 MB)。

## 3. 机上の予算(1024²、M6 mini、M8 §2 と M9c の実測から)

1 ブロック(いま、計算 281.5 ms):

| | ms |
|---|---|
| attention の matmul(wq / gate / wo 3×18.4、wk / wv 2×4.8) | 65 |
| sdpa | 27 |
| **MLP**(gate / up 2×48.9、down K4 51) | **149** |
| 要素演算(norm の bf16 化後) | 約 24〜34 |
| 重み(q8) | attention 141 MB、**MLP 320 MB**、計 461 MB |

MLP の FLOP は 2.49 TFLOP/ブロック。ANE の実効を R TOPS、ANE の取り分を a として
GPU 側 (1−a)×149 ms と ANE 側 a×2490/R ms が釣り合う点:

| ANE の実効 | a | MLP の相 | 1 ブロック | 1 ステップ | ANE 常駐(a×320 MB×28) | 期待 footprint |
|---|---|---|---|---|---|---|
| いま(GPU のみ) | 0 | 149 | 281 | **7.9 s** | 0 | 6.2 GB |
| 22 TOPS(ANE 1 基、Draw Things の実測) | 0.57 | 64 | 約 190 | **約 5.5 s** | 5.1 GB | 約 11 GB |
| 44 TOPS(デュアルが公称通り) | 0.72 | 41 | 約 167 | **約 4.9 s** | 6.5 GB | 約 12.5 GB |

- 上限は **1.5〜1.7 倍**。attention の間(92 ms)は ANE が遊ぶので 2 倍にはならない。
- 受け渡し 100 MB/ブロックのオーバーヘッドは含んでいない(§4-5 で測る)。
- footprint 11〜12.5 GB は 16 GB 機で **かなり際どい**(M7 で claimable を測っている。VAE の 14 GB は DiT 破棄後なので別)。
  a を下げれば常駐は減るが速度も減る。**a は flag で可変にし、メモリで決める。**
- M3 Pro は GPU 5.6 TFLOPS に対し ANE 16 コアなので比率が逆転する(TTS はそこで 1.35 倍)。
  ただしこの計画の対象は M6 mini で、M3 Pro は「遅くならない」だけ確認する(既定 off)。

## 4. probe で決めないと進めない未知数

1. **ANE の実効 TOPS が Krea 2 の形で出るか。** 4126×6144 @ 6144→16384 と 16384→6144。TTS の DiT より
   1 桁大きく、K=16384 は ANE の上限に近い可能性がある。M1 では形によってコンパイルが黙って失敗し
   CPU に落ちた(`17-m1-ane-factors.md` 2-2、9 倍遅い)。**unified log の "Model load failed" を必ず見る。**
2. **デュアル ANE を 1 モデルで束ねるか、2 モデルで使い分けるか。** どちらでも a は変わるだけだが実効が違う。
3. **GPU と同時に回して両方が落ちないか。** メモリ帯域とクロックは共有。`ceiling_probe.py` と並走で測る。
4. **常駐メモリ、初回コンパイル、2 回目のロード時間。** TTS では 1 package 0.2 s(OS がキャッシュ)。28 ブロック分。
5. **呼び出し 1 回のオーバーヘッド。** 50 MB 入 + 50 MB 出、`shared_memory` 経由、ブロックあたり 1 回。5 ms 以内が目安。
6. **fp16 活性の数値。** ANE は bf16 を受けない。実ブロックの MLP 入力と `silu(gate)*up` の max abs を全 28 ブロックで
   取り、65504 に対する余裕を見る。出力差は M9b §5 と同じ表(max / mean / 1 ULP 超の割合)。
7. **重みの再量子化。** GPU は affine q8(group 64、scale + bias)、Core ML の int8 は行ごと、または blockwise の
   scale + zero point。同じ 8bit でも丸めが変わる。**q8 固定の範囲内だが、画像で判定する。**
   int8 の活性(W8A8)は ANE が速い可能性があるが、画像判定に通ったときだけ。

## 5. 段階(M6 mini の実ウェイトで測る)

### M11a. probe `tools/bench/ane_probe.py`(半日) ← ここから

- `coremltools` で Linear を mlprogram に変換(`compute_units=ALL`、fp16 入出力)。形は 3 つ
  (6144→6144、6144→16384、16384→6144、M=4126 と 6430)。変種: (i) int8 行スケール + fp16 活性、
  (ii) int8 group-64(`block_size`)+ fp16 活性、(iii) int8 活性(できれば)。
  重みは実 block 0 を `mx.dequantize` して再量子化する(合成ではなく実データ)。
- 単体: ウォームアップ 3、10 回平均で TOPS。初回コンパイルと 2 回目のロードの時間。
  worker の `phys_footprint`。unified log で ANE に載ったことを確認(TTS の判定コードを写す)。
- デュアル: 同じ package を 2 プロセスで同時に回してスループットの合計を見る。1 プロセスで倍になるかも見る。
- 同時: `ceiling_probe.py --only`(6144→16384 の形)を並走させ、GPU と ANE の両方の TOPS を単体と比べる。
- 受け渡し: `shared_memory` 経由で 50 MB 入 / 50 MB 出を含めた 1 呼び出しの時間。
- 数値: `block_budget.py` の経路で実 block の MLP 入力を dump し、fp16 で通した出力と bf16 GPU の差。
- 記録は `docs/16gb/measurements/2026-09-22-m11a-ane-probe.md`。

**合格基準(全部)**: fp16 活性で ANE が **GPU 併走下でも 15 TOPS 以上**、GPU 側の低下 5% 以内、
呼び出しオーバーヘッド 5 ms 以内、常駐が §3 の見積りの ±20%、fp16 の出力差が norm の bf16 化(M9b §5)と同程度。
**1 つでも外れたら「M6 でも ANE 併用は割に合わない」と数字つきで閉じ、M11b には進まない。**

### M11b. MLP の列分割を本番に(M11a 合格のときだけ)

- `feed_forward.py`: `down_splits` と同じ流儀で `ane_share`(a)を持ち、GPU 側は列の (1−a) だけを計算し、
  ANE の部分積を足す。常駐モードでは常に 0。
- `krea2_weight_stream.py` / `Krea2BlockStream`: ANE の取り分の重みは GPU 側で読まない(ブロック 461 → 461 − a×320 MB)。
  MLP 入力を worker に送るのは attention の `mx.eval` の直後、GPU 側の gate/up を発行してから受け取る。
- 新規 `src/mflux/models/krea2/model/krea2_ane_worker.py`: `irodori_tts/ane_worker.py` を写す(spawn、`shared_memory`、
  ANE 失敗の検出と MPS ならぬ GPU への切り戻し)。
- 前処理 CLI: 28 ブロック分の Core ML package を 1 回作る(`tools/bench/` から昇格する `bake` 系と同じ扱い)。
  出力先は `~/Library/Caches/mflux/16gb-bench/krea2-lowram/ane/`。
- 有効化は `--block-streaming --ane-share 0.57` のような flag。既定 off。
- 測定: `stream_ab.py` で本番 A/B(1024² / 1280²)、`swapwatch.py` で clean、画像判定。M3 Pro は既定 off の回帰だけ。

**合格基準**: 1024² **6.0 s/step 以下**、clean(swapouts 0)、1280² も clean、画像判定合格。

### M11c. 実行時 LoRA との共存

M10 の LoRA は gate / up / down の side path。ANE の列に対応する delta は rank 64 なので GPU で計算して
足す(`_down_adapter` の形をそのまま使う)。M11b の後。

## 6. やらないこと

- DiT 全体を ANE で回す(TTS で遅かった。Draw Things も M5 では GPU より遅いと明言)。
- attention を ANE に載せる(sdpa は天井、行列積ではない)。
- ANE でメモリを減らす期待(TTS で 0 バイト。この計画では**増える**)。
- 画質を落とす手。int8 活性は画像判定に通ったときだけ。

## 7. 出典

- Irodori-TTS: `~/dev/Irodori-TTS/docs/note-mac.md`(ANE + GPU 並走 1.35 倍、メモリ不変、GIL)、
  `docs/experiments/17-m1-ane-factors.md`(形と ANE コンパイル失敗の検出)、`irodori_tts/ane_worker.py`
- [Draw Things: Making Apple Neural Engine work in a custom inference stack](https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in)
  (行列積だけを Core ML に、8-bit S、macOS 26 で int8 配列、M4 1.8 倍、実効 22 TOPS / 公称 38)
- [Draw Things Integrates Apple Neural Engine into Runtime](https://letsdatascience.com/news/draw-things-integrates-apple-neural-engine-into-runtime-f78ca45e)
  (M5 / A19 Pro で ANE+NAX hybrid が既定)
- [Draw Things: Metal FlashAttention v2.5 w/ Neural Accelerators](https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural)(GPU 側の neural accelerator。NAX はこれ)
- [Apple: M6 と M5 Ultra](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/)(デュアル 16 コア Neural Engine、ピーク 2 倍)
- [M6 vs M5: Dual Neural Engine](https://www.ithinkdiff.com/m6-chip-vs-m5-chip-2nm-design-12-cores-and-dual-neural-engine/)

## 8. 再開手順(2026-09-22 中断時点)

1. mini への経路: Thunderbolt bridge(`m6-tb`)が落ちていたので **`ssh m6-lan`**(192.168.0.64)を使う。
   `just studio-mini` の rsync は `m6-tb` 前提なので、手で同期する:
   `rsync -a --exclude .venv --exclude .git --exclude '*.pyc' src tools pyproject.toml uv.lock m6-lan:dev/mflux/`
2. mini は Python 3.14 / coremltools 9.0 が `uv run --with coremltools` で入る(確認済み、importOK)。
   studio(pid 6282、19 MB、idle)が動いているが GPU は使っていないので止めなくてよい。
3. 最初の煙試験(mini で):
   `source ~/.local/bin/env; cd ~/dev/mflux && uv run --with coremltools python tools/bench/ane_probe.py --only convert,single --shapes gate --variants row --m 4126`
   - まず `Block0` の key 名(`attn.wq.weight` 等)が `Krea2BlockStream.read(0)` の flatten と合うか。合わなければ `Block0.weights` を直す。
   - `ct.convert` の fp16 I/O が MIL Program 入力で通るか(通らなければ fp32 I/O に落ちて `io` に理由が出る)。
   - `[linear->NeuralEngine x1]` が出るか。出なければ `--layout conv` を試す。
4. 全部: `--only convert,single,dual,concurrent,handoff,numerics --json docs/16gb/runs/m6mini/<ts>-m11a-ane-probe.json`
   → 続けて `--only actstats --json docs/16gb/runs/m6mini/<ts>-m11a-actstats.json`(実走 1 枚、約 40 s)。
5. 記録は `docs/16gb/measurements/2026-09-22-m11a-ane-probe.md`(M9 の書式)。合格基準は §5 M11a。
