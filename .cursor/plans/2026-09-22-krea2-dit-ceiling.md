# 計画: Krea 2 の DiT を M6 mini の天井(6 s/step)に近づける — 測定と改善

- 日付: 2026-09-22
- ブランチ: `feat/krea2-block-streaming`
- 前提となる実測: [M9: この GPU の天井と MLX が届いていない場所](../../docs/16gb/measurements/2026-09-22-m9-ceiling-probe.md)、
  [M8: 1 ブロックの内訳](../../docs/16gb/measurements/2026-09-22-m8-block-profile.md)、
  [M8c/d: K 分割の敗因と cache limit](../../docs/16gb/measurements/2026-09-22-m8c-m8d-kslices-cache-limit.md)
- 前の計画: [DiT を 2〜3 倍速くする](2026-09-22-krea2-dit-speed.md)(M8a〜f 済、21.2 → 9.26 s/step)
- 管理先: fork [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)(upstream へ PR は出さない。MLX への issue は別)

## 0. 前提(動かせない要件)

前の計画と同じ。**q8 固定**、**スワップは失敗**、`tools/swapwatch.py` 越し、対象構成は
q8 + 4step LoRA / 1024² と 1280² / 4 ステップ / euler / guidance 1.0 / seed 42。
画質を落とす手は使わない。**出力の数値が動く変更は画像で判定する**(同じ seed を並べ、
ピクセル差の統計を添える)。M3 Pro 18GB でも遅くならず footprint が増えないこと。

## 1. 問いへの答え(M9 で確定)

**「発売日の M6 で 9.26 s/step は遅すぎないか」** → 半分正しい。

- **シリコンの天井は 19 TFLOPS。** MLX の密 bf16(4096³)19.2、PyTorch MPS 18.7〜19.0 で一致。
  MPS は Apple 自身の Metal カーネルで neural accelerator も使うので、これ以上は誰も出していない。
  Apple の「M5 比 +30%」、Geekbench Metal +34% と矛盾しない。
- **1 ステップは 112 TFLOP(1024²)。19 TFLOPS で 5.9 s、ブロック外を足して 6.2 s が下限。**
  9.26 s は下限の 1.5 倍。うち **1.3 s は明らかな異常**で、`mlp.down`(16384→6144)の MLX
  カーネルが MPS の半分の速さ(9.1 vs 17.9 TFLOPS)で走っている。0.32.2 でも直っていない。
  残りは未融合の要素演算(0.5〜0.7 s)と、MLX の q8 カーネル自体が天井より 1 割遅い分(0.5 s)。
- **q4 / q6 に落とす価値はない。** 崖の外で q4 は q8 の +7%。崖の中では +60% だが、崖は K 分割で
  消せる(15.7 TFLOPS)ので、崖を直したあとに q4 が買うのは 7%(0.5 s/step)。
  I/O は既に隠れているので重みが半分になってもステップ時間は変わらない。**q8 のまま。**
- **ComfyUI(MPS)に移っても速くならない。** MPS は崖がない代わりに毎ステップ逆量子化を払い
  (11 ms / 100 MB。崖の形で逆量子化込み 61 ms 対 MLX 93 ms、速い形では MLX が勝つ)、
  ブロック合計はほぼ同額。attention は同速。そして**ストリーミングがないので 16GB に載らない**。

現実的な到達点は **7.5 s/step 前後(1.25 倍)**。それ以上は MLX のカーネルが MPS に追いつく
かどうかで、fork の外(§M9c-4)。

## 2. 1 ブロック(1024²、323 ms)の予算と狙う額

| | いま | 狙い | 段 |
|---|---|---|---|
| `mlp.down` 16384→6144 | 93 ms | 46〜53 | **M9c** |
| 要素演算(norm ×2、変調、残差、silu×up、RoPE、QK norm) | 約 30 | 10〜15 | **M9d** |
| プリフェッチの汚染 | 3〜5(1280² で 32) | 0〜3 | M9e |
| 7 つの速い matmul + sdpa | 190 | 175(MLX 側) | M9c-4 |
| ブロック外(txtfusion / first / last / サンプラ) | 0.22 s/step | — | 測るだけ |

## 3. 段階(すべて M6 mini の実ウェイトで測る。M3 Pro は最後に回帰)

### M9a. 揃えて再現する ← 済(9.01 s/step、M8d と 0 ピクセル差。[M9b](../../docs/16gb/measurements/2026-09-22-m9b-block-budget.md) §1)

- mini の作業ツリーは `dcacf52` + 未コミット。M3 Pro の HEAD `5ca3c59` に揃える
  (`git stash` → `git pull fork`、または rsync)。`git diff` が空であること。
- README §実行のしかた を 1024² で 1 回。**9.26 s/step ± 2%、footprint 5.72 GB、clean** を再現
  してから始める。ずれたら別プロセスが GPU を使っている(`ps aux | grep python`)。
- 天井の測定は済(M9)。`ceiling_probe.py` は温度の目安として各段の前後に `--only sdpa` だけ回す
  (24.2 ms から動いたらクロックが落ちている)。

### M9b. 323 ms の内訳を 1 つのプロセスで確定する ← 済(matmul 単体和 256 + sdpa 27 + 要素演算 34。「鎖の隙間 40 ms」は合成鎖の artefact。[M9b](../../docs/16gb/measurements/2026-09-22-m9b-block-budget.md))

M8 の内訳は別スクリプトの数字の寄せ集めで、`matmuls` の鎖が個別合計より 40 ms 遅い件が
未解明のまま。`block_profile.py` を拡張して、実 block 0 の bf16 で:

1. 8 本を **1 本ずつ足していく鎖**(wq → +gate → +wk → … → +down)を測り、どこで
   個別合計から離れるかを見る。離れる場所が中間活性 135 MB(4126×16384)の確保なら M9d の
   compile で消えるはず、matmul 間の依存待ちなら消えない。
2. `mx.compile` した block と素の block を並べる(M9d の見込みを先に出す)。
3. 要素演算を bf16 のまま通した版(RMSNorm / QK norm の float32 往復なし)を並べる。
4. `MTL_CAPTURE_ENABLED=1` で block 1 回を capture し、`nax_probe.py` の方法でカーネル名の
   一覧を取る。**目的はカーネル数**(要素演算がいくつのディスパッチに分かれているか)。
   時間はトレースからは読めないので 1〜3 で測る。

合格基準: 「down 93 / 7 本 163 / sdpa 27 / 要素 X / 依存待ち Y」で X + Y ≈ 40 と説明が閉じる。

### M9c. `mlp.down` の崖を消す ← 済(**採用**: 直接読み + K4 バッチ qmm。M8c の敗因は `async_eval` が呼び出し側を 100〜250 ms 止めること。1024² 9.01 → **7.90 s/step**。[M9c](../../docs/16gb/measurements/2026-09-22-m9c-prefetch-interference.md))

M8c の K 4 分割は単体 −30 ms、本番(プリフェッチあり)+42 ms で不採用になった。
M9 §2 で崖が**帯域律速**(ビット数に比例して速くなる)と分かったので、敗因は
「K=4096 の 4 本はキャッシュに乗って速いが、裏の 461 MB の memcpy がそれを追い出す」
と読める。順に試す(各 1 回、1024² 本番、交互測定 1→4→1→4):

1. **プリフェッチを別スレッドに**(M8b 案 B)+ K 4 分割。干渉が主スレッドの `mx.eval` による
   スケジューラの塞がりなら消える。最も安いので先。
2. **干渉源の切り分け**: プリフェッチ元を (a) 温かいページキャッシュ(SSD DMA なし、memcpy のみ)、
   (b) 冷え、(c) `mx.load` を使わず `os.pread` で読んで `mx.array` にする(コピー回数を 1 減らす)、
   で +42 ms がどう動くか。**この結果で 3 か 4 を選ぶ。**
3. **読みの位置替え**: プリフェッチを 2 片に割り、attention 半分の裏と `mlp.gate/up`(K=6144、
   崖なし、98 ms)の裏に置いて、**K 分割した `down` は読みが終わってから走らせる**。
   `Krea2StreamedBlock` が block の forward をサブモジュール単位で呼ぶ形になる
   (`SingleStreamBlock.__call__` は 2 行なので写せる)。1280² は attention が長いので楽。
4. **MLX に報告する**(fork と独立、今日中に出せる): M9 §1〜2 の表(K スイープ、gs / dtype / M
   不変、0.32.2 でも同じ、密 bf16 8192³ も 12 TFLOPS、**MPS は同じ形で 17.9**)と
   `ceiling_probe.py --only ksweep` を最小再現として添える。密 GEMM も落ちるので
   `steel`/`nax` の GEMM の K ループ(split-K または BK)の問題として出す。

合格基準: 本番 1024² **≤ 8.2 s/step**、clean、footprint +0.3 GB 以内、画像判定
(K 分割は積算順が変わる。M8a と同形式でピクセル差の統計)。
1〜3 のどれも本番で勝てなければ「MLX 側の修正待ち」と数字つきで結論し、4 の issue 番号を記録する。

### M9d. 要素演算を融合する ← 済(**norm の bf16 化だけ採用** −10 ms。`mx.compile` はブロック全体でも部分でも遅くなり不採用。M9b §4 / §4b)

1. **`Krea2RMSNorm` の float32 往復をやめる。** `mx.fast.rms_norm` は内部で float32 に累積する
   ので `x.astype(float32)` は二重。まず実データ(block 0 の入力)で bf16 入力と float32 入力の
   出力差(max / mean)を出し、bf16 の丸め 1 ULP 以内なら採用。M8 §3 の 3.9 → 0.9 ms ×2、
   QK norm 4.8 → 約 1.5。
2. **内側の `SingleStreamBlock` を `mx.compile`。** ストリーミングの wrapper(ディスク読みと
   `mx.eval`)は compile の外に置き、block 本体だけを 1 回トレースする。28 ブロックが同じ形なので
   **module を 1 個だけ持ち、`update` で重みを差し替え**、`mx.compile(fn, inputs=block.state)`
   で状態を入力として追わせる(28 回トレースしない)。1024² と 1280² で形が違うので 2 回
   トレースされるのは許容。融合の対象: 変調 `(1+scale)*n+shift`、残差 `x + gate*y`、
   `silu(gate)*up`、RoPE の float32 往復、`sigmoid(gate)` の積。
3. `mx.repeat`(GQA)を消して sdpa に 12 ヘッドのまま渡す(1.2 ms。M8a で計画して未実施)。

合格基準: ブロック −15 ms 以上、footprint 不変、画像判定(1 は数値が動く可能性、2 は動かない
はず → 動いたらバグ)。**M9c と別々に測る**(M8c の教訓: 単体で速い変更が本番で遅いことがある)。

### M9e. プリフェッチの汚染 ← M9c で解消(1280² の計算 573 → 512 ms。読みは主スレッドを離れないスレッドの preadv になった)

M8b §2 の 1280² +32 ms/ブロック。M9c-2 の切り分けと同じ道具で:
温かいページキャッシュ vs 冷え、スレッド化、`os.pread` + `F_NOCACHE`、読みを 2 片に割る。
1024² では 3〜5 ms なので **1280² だけで判定**。合格基準: 1280² の計算 / ブロック 573 → 550 ms 以下。

### M9f. 持続クロック(ユーザ操作。sudo が要る)← 未実施

```sh
sudo powermetrics --samplers gpu_power -i 1000 | grep -E "GPU HW active frequency|GPU Power"
```

を 1280² の本番と並走させ、ステップごとの `compute_ms`(15.50 → 15.95 s、+3%/step)と並べる。
落ちていれば「mini の持続性能はこれ」と記録するだけ(対策はしない。前計画 §M8e)。
`ceiling_probe.py --only sdpa` を本番の直前・直後に回す代替計測も併記する。

### M9g. M3 Pro で回帰 + `iogpu.wired_limit_mb=0` 再測定 ← 速度は 22.89 s/step(M8f 22.96、±0)、footprint 5.04 GB。ただし作業機で常駐アプリありの条件で swapouts 8 ページ(clean でない)。**常駐アプリを落として再測定**(M9c §5b)

M9c/d を入れた状態で M3 Pro の 1024² を 1 回。22.96 s/step より遅くならず、footprint 4.66 GB
より増えず、clean。同時に `sudo sysctl -w iogpu.wired_limit_mb=0` で HANDOFF §6 の宿題を消す。

## 4. 到達点(1024²、M6 mini、1 ステップ)

| 段 | ブロック | 1 ステップ | 対いま | 実測 |
|---|---|---|---|---|
| いま(M8d) | 323 ms | **9.26 s** | 1.00 | 9.01(M9a、同日再測定) |
| + M9c(崖) | ≈ 280 | **≈ 8.0 s** | 1.16 | |
| + M9d(融合) | ≈ 260 | **≈ 7.5 s** | 1.24 | **281.5 ms / 7.90 s**(M9c + norm bf16。融合は不採用) |
| 天井(MLX が MPS に追いつく) | 213 | ≈ 6.2 s | 1.49 | |

1280²: 15.88 → **14.36 s**(実測)。M3 Pro: 22.96 → 22.89(±0)。

生成全体(TE 2 s + DiT + VAE 6 s)は 1024² で 45 → 約 40 s(DiT 4 × 7.9 = 31.6 s)。

**これ以上を望むなら機材の話になる**: 1 ステップ 112 TFLOP は固定で、19 TFLOPS の GPU では
6 s を切れない。q4 でも 7%、MPS でも同額。

## 5. 触るファイル

| 段 | ファイル |
|---|---|
| M9b | `tools/bench/block_profile.py`(鎖の段階測定、compile 版、bf16 norm 版) |
| M9c | `models/krea2/weights/krea2_weight_stream.py`(スレッド化 / 2 片プリフェッチ / サブモジュール単位の forward)、`feed_forward.py`(K 分割の置き場) |
| M9d | `model/krea2_transformer/common.py`(RMSNorm)、`attention.py`(repeat)、`krea2_weight_stream.py`(共有 module + compile) |
| 計器 | `tools/bench/ceiling_probe.py`、`tools/bench/mps_probe.py`(済) |
| 記録 | `docs/16gb/measurements/2026-09-22-m9*.md`、`runs/m6mini/`、`images/` |

## 6. 測定の作法(この計画で足すもの)

- **MPS を物差しにする。** MLX の数字が「遅い」かは MPS の同形で判定する。torch は
  `uv run --no-project --isolated --with torch --python 3.12` で使い捨て環境に入れ、
  mflux の環境には入れない。
- **崖は帯域律速。** 崖の形の測定は他の重い読み書き(プリフェッチ、別プロセス)を止めて取る。
  本番形(プリフェッチあり)での測定と単体測定は別々に記録する。
- **同期してから測る。** mini と M3 Pro の作業ツリーが違う状態で数字を比べない(M9 §8)。
- 前計画 §5 はそのまま(GPU 独占、scales の dtype、冷え / 温まり、画像判定の手順)。

## 7. 非目標

- **q4 / q6 / mxfp8 への変更**(M9 §4 で計算効率上の理由も消えた)。
- **ComfyUI / PyTorch MPS への移行**(M9 §6)。MPS は物差しとしてだけ使う。
- 逆量子化して密行列積にすること(6144→16384 で 17.0 → 10.4 TFLOPS に落ちる)。
- text encoder / VAE の高速化(合わせて 8 s)。
- ファン制御、`powermetrics` の結果に対する対策。
- upstream mflux への PR。
