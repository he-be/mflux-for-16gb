# 引き継ぎ(2026-09-22 時点)

新しいセッションはこのファイルだけ読めば再開できる。次に読むのは
[計画](../../.cursor/plans/2026-09-22-krea2-block-streaming.md)。

## 1. やろうとしていること

M3 Pro 18GB の MacBook Pro で **Krea 2 Turbo q8 + 4step 蒸留 LoRA** の画像生成を、
**スワップさせずに**通す。その後 M6 mac mini でも同じ構成を回す。
**両方とも済んだ**(M6 / M7)。

### 動かせない前提

- **q8 固定。** 8bit 未満は画像として破綻しないが、描き込みが減り表現が変わることが
  CUDA 側の検証で確定している。**q4 / q3 に落とす案は検証しない。**
- **スワップは失敗。** 実行は `tools/swapwatch.py` 越し。`clean` 以外は不採用。
- **「載らない」は正当な結論。** 無理やり載せて駄目なら数字つきでそう報告する。
  画質を落とす代案を出さない。
- 対象構成: q8 + 4step LoRA / 1024² / 4 ステップ / `--scheduler euler` / guidance 1.0 / seed 42。

## 2. 場所

| 何 | どこ |
|---|---|
| 作業リポジトリ | `/Users/mh/dev/mflux`(upstream `mflux-community/mflux` の clone) |
| **M6 mac mini(16GB)** | ssh `m6-tb`(Thunderbolt bridge 169.254.24.129)。`~/dev/mflux` と `~/Library/Caches/mflux/16gb-bench/krea2-lowram` に同じものが入っている。uv は `~/.local/bin/env` を source してから |
| fork remote | `fork` = `https://github.com/he-be/mflux-for-16gb`(**PR は出さない**) |
| ブランチ | `feat/krea2-block-streaming` |
| q8 ウェイト | `~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/ad5c0b1784c486bd45c2ced8a9f10aa45293a7c8/` |
| 4step LoRA | `~/Library/Caches/mflux/loras/krea2_turbo_4step_rank_64_lora_comfyui.safetensors` |
| 中間生成物 | `~/Library/Caches/mflux/16gb-bench/`(埋め込み・latent・**q8 TE 4.3GB** と**焼き込み済み DiT 13GB**。リポジトリ外) |
| スワップ見張り | `tools/swapwatch.py` |
| 計測スクリプト | `tools/bench/`(`memstat.py` / `te_encode.py` / `dit_steps.py` / `vae_decode.py` / `block_stream.py` / `bake_lora_checkpoint.py` / `quantize_te_checkpoint.py` / `matmul_probe.py` / `nax_probe.py` / `qmm_spy.py`) |
| 文書 | `docs/16gb/`(research / measurements / runs)、計画は `.cursor/plans/` |

ダウンロードは完了済み(q8 21GB + LoRA 438MB)。再取得は不要。LoRA は
`lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors`
という指定でも通る(mflux が自前のキャッシュに写しを持っている)。

## 3. 確定した事実(断りがなければ M3 Pro 18GB での実測。M6 mini の数字は §5)

### ハード

```
hw.memsize                        19.33 GB
max_buffer_length                  9.66 GB
iogpu.wired_limit_mb              既定 0(単位は MiB、再起動で 0 に戻る)
  └ 0 のとき   working set 12.88 GB / 14336 MiB → 15.03 GB / 15360 MiB → 16.11 GB
mx.set_wired_limit                既定 0(MLX は何も wire しない)
システムの wired                   約 2.0 GB
```

**`iogpu.wired_limit_mb` の既定は 0。** 以前の引き継ぎが 14336 を既定として記録して
いたが、あれは前セッションが `sysctl` で手動設定した残骸だった。

### q8 チェックポイントのサイズ

| | |
|---|---|
| DiT 全体 | 13.62 GB(blocks 12.92 + globals 0.71) |
| 1 ブロック | 461.3 MB(28 ブロックすべて同一、29 テンソル) |
| 4step LoRA | 0.44 GB(bf16, rank 64) |
| text encoder (Qwen3-VL-4B, bf16) | 8.05 GB |
| VAE | 0.51 GB |
| 全部同時 | 22.2 GB → **載らない** |
| **DiT(LoRA を bake した場合)** | **13.62 GB**(bake 中だけ 15.23 GB の一時ピーク) |
| DiT(bake しない場合) | 14.06 GB |
| + 1024² のアクティベーション | **+1.94 GB → 要求 15.96 GB** |

### 読み出し(実シャード、`F_NOCACHE`)

| | |
|---|---|
| 1 ブロック(29 テンソルを拾い読み) | **74 ms = 6.24 GB/s** |
| DiT 全体 | 3.05 s = 4.47 GB/s |

シャード内でブロックのテンソルは連続していない(block 15 は 2.09GB のシャードの
1.37GB に散らばる)が、拾い読みでも帯域は落ちないので並べ直しは不要。

### 計算(合成テンソル、実ブロックではない)

| | |
|---|---|
| bf16 matmul 5120×6144@6144×6144 | 65.5 ms = 5.90 TFLOPS |
| q8 / q4 quantized matmul 同形状 | 75 ms = 5.1 TFLOPS |
| 合成 28×138MB の bind→eval→drop | 常駐 0.17GB、計算律速のまま |

同じ形を M6 mini で測ると bf16 12.35 / q8 **17.16** / q4 18.49 TFLOPS。
**q8 が bf16 より速いのが neural accelerator の指紋**(M3 Pro では逆に遅い)。
再現は `tools/bench/matmul_probe.py`。

### 計器(ここを間違えると全部無意味になる)

- **`ps rss` は MLX の確保を 1 バイトも見ない。** 4.00 GB の `mx.array` を持つプロセスを
  `ps` は 0.03 GB と報告する。`phys_footprint` なら 4.10 GB。
- **`footprint -p` は大きなプロセスを整数 GB に丸める。** これに引っかかって
  「このマシンの上限は 16.12 GB」という**誤った結論を一度出した**(13326 / 14350 /
  15374 は 13/14/15 GB + launcher の 14 MB だった)。`proc_pid_rusage` を
  ctypes で直接読むこと(`swapwatch.py` は修正済み)。
- 判定に使うのは `mx.get_peak_memory()`(プロセス内)と `phys_footprint`(外から、
  ツリー全体を合計)。**2 本が一致することを毎回確認する。**
  DiT では一致する(15.57 / 15.96 GB)が、**VAE では一致しない**
  (4.40 / 14.16 GB。畳み込みが MLX の外側で確保している)。
- 空きメモリは `free` ではなく **claimable = free + inactive + speculative + purgeable**
  で見る(`tools/bench/memstat.py`)。
- 詳細: [計測の土台](measurements/2026-09-22-memory-instrumentation.md)

### M2 の結果(合格)

TE だけのプロセス: MLX ピーク **8.19 GB**、footprint ピーク 8.11〜8.29 GB、
**swapouts 0**、verdict `clean`。実体化 1.3〜2.1 s、エンコード 0.63〜0.82 s。
埋め込みは `(1, 30, 30720)` bf16 = 1.84 MB。
再起動の前後で 4 回測って数字は同じだった。
[記録](measurements/2026-09-22-m2-text-encoder.md)

### mflux 側の作法(コードを読んで確認済み)

- `--steps 4` を明示(既定 8)、`--scheduler euler`(既定 er_sde)、guidance は 1.0 のまま
  (mflux の 1.0 = 単一パス = ComfyUI の cfg 1.0)。
- LoRA の全 456 キーが `Krea2LoRAMapping` に 456/456 で一致(検証済み)。
- q8 なら LoRA は bake してよい(8bit 未満のときだけ `--no-bake-lora` が要る)。
  **このマシンでは bake すべき**(実測: 常駐 -0.44 GB、1 ステップ 41.9 → 30.9 s)。
  bake すると LoRA は q8 のウェイトに畳み込まれ、常駐は 13.62 GB のまま増えない。
- σ スケジュールは動的シフト。LoRA の学習点 (μ=1.15) と一致するのは **1280×1280**。
  1024² は μ=0.906 でわずかにずれる。1024²/4 ステップの σ は
  `[1.0, 0.8813, 0.7122, 0.4521, 0.0]`(実測)。
- `Krea2Initializer.init` は既定では 3 コンポーネントを構築して `mx.eval(model)` で
  一括実体化する。**`--block-streaming` のときだけ何も構築せず、`Krea2StagedLoader` が
  生成中に 1 個ずつ作って手放す**(M6)。`tools/bench/` も同じ経路
  (`WeightLoader.load_single_local`)を使う。
- q8 DiT の配線は検証済み: 956 パラメータが 956/956 で一致、形の不一致 0、
  量子化層 263、`bits=8`。
- **`mx.load` は lazy。** 上の検証は 13.62 GB を一切実体化せずに通る(active 0.000 GB)。
  読み出し時間を測るなら `mx.eval(model)` を明示すること。
- `MemorySaver` はエンコード後に TE を、`--low-ram` はループ後に DiT を破棄する
  (`memory_saver.py:77`)。

## 4. 結論: **目標構成は 18GB 機でスワップなしに通る。mflux 本体に入っている**

方法は**ブロック単位ストリーミング**(M5)。常駐方式(M3)は不可能だった。
**M6 で mflux 本体に入り、`--block-streaming` 1 本で通る。**
1280²(LoRA の学習 σ と一致する解像度)も clean。

**生成のたびに走る段**(どの段も 5 GB を超えない):

| 段 | mx peak | 実 footprint | 時間 | verdict |
|---|---|---|---|---|
| text encoder (q8) | 4.47 GB | **4.96 GB** ← 最大 | 2 s | **clean** |
| **DiT ストリーミング** | 3.21 GB | 3.72 GB | **119 s**(4 × 29.8) | **clean** |
| VAE(タイル 256) | 2.98 GB | 4.28 GB | 5.6 s | **clean** |

**1 回だけ必要な前処理**(結果は `~/Library/Caches/mflux/16gb-bench/` に置いてある):

| | mx peak | 時間 | 出力 |
|---|---|---|---|
| TE の量子化 `quantize_te_checkpoint.py` | 1.19 GB | 2.3 s | `krea2-te-q8/` 4.27 GB |
| LoRA の焼き込み `bake_lora_checkpoint.py` | 1.89 GB | 13 s | `krea2-q8-4step-baked/` 13 GB |

q8 + 4step LoRA / 1024² / 4 ステップ / euler / guidance 1.0 / seed 42。
出力は `docs/16gb/runs/images/`。**常駐版とビット単位で一致**している
(mflux CLI 経由の M6 の出力も、`tools/bench/` 版と**ピクセル完全一致**)。

### 各段の結果

| 段 | 結果 |
|---|---|
| M1 ベースライン | 済。`iogpu.wired_limit_mb` の既定が 0 と判明 |
| M2 TE | **合格 clean** |
| M3 DiT 常駐 | **不合格**。要求 15.96 GB。4 ステップは完走するがスワップする |
| M4 VAE | **合格 clean**。タイル 256 |
| M5 ストリーミング | **合格 clean**。要求 3.72 GB、比 13.65、ビット一致 |
| M5b TE を q8 に | **採用**。8.29 → 4.96 GB。画像は目視で同じ、描き込みは同等 |
| M6 本体への実装 | **合格**。`--block-streaming`。CLI 経由で 29.60 s/step、比 13.59、**出力はピクセル完全一致**。1280² も clean(48.91 s/step、6.89 GB) |
| M7 M6 mac mini 16GB | **合格 clean**。1024² 20.9 s/step / 5.32 GB、1280² 35.21 s/step / 6.46 GB、swapouts 0、`iogpu.wired_limit_mb` は既定 0。**neural accelerator が効いている** |

### M3(常駐)がなぜ駄目だったか

ウェイト 13.63 + アクティベーション 1.94 + 0.35 = **15.9 GB**、
システムの 2.0 GB と合わせて **17.9 / 19.33 GB**。ページキャッシュが 0.15 GB まで
削られてもなお足りず、毎秒数百ページ掃き出す。768² に落としても消えない。

### M5(ストリーミング)がなぜ通るか

28 ブロック(各 461.3 MB)をディスクに置いたまま、1 個ずつ bind → forward → drop。

```
常駐            : globals 0.706 GB のみ
I/O             : 71.8 ms / ブロック(M0 の実測 74 ms と一致)
計算            : 984.7 ms / ブロック
計算 ÷ I/O      : 13.65   ← 合格基準 2.0
1 ステップ      : 29.8 s(常駐版 28.2 s の +5.8%)
```

**mflux 本体は 1 行も変えていない。** `Krea2Transformer.__call__` が
`for block in self.blocks:` なので、`transformer.blocks` をラッパのリストに
差し替えるだけで成立する。

### 効いた手・効かなかった手(全部実測済み。もう一度試す必要はない)

| 手 | 結果 |
|---|---|
| **ブロック単位ストリーミング** | **これが答え。** 15.96 → 3.72 GB |
| **LoRA を事前にチェックポイントへ焼き込む** | 実行時コスト **ゼロ**。焼き込み自体も 2.14 GB / 13 秒 |
| VAE のタイル 256 | 実 footprint 14.15 → 4.28 GB、代償 0.9 秒 |
| `mx.set_wired_limit` | 常駐方式には**必須**だった。ストリーミングでは**不要** |
| `sudo sysctl iogpu.wired_limit_mb` | 同上。ストリーミングでは**不要** |
| `mx.set_cache_limit` を絞る | **効かない。**ロード中は無関係、ループ中は 9 倍悪化 |
| 解像度 768² | 常駐方式では速度は倍になるがスワップは消えない |
| `F_NOCACHE` でページキャッシュ迂回 | **不要。**OS は既に正しく回収している |

## 5. M7 の結果(M6 mac mini 16GB)← 済

**16GB 機で通った。§1 の目的は達成。** 詳細は
[M7](measurements/2026-09-22-m7-m6-mac-mini.md)。

| | M3 Pro 18GB | **M6 mini 16GB** |
|---|---|---|
| 1024² 1 ステップ | 29.60 s | **20.9〜21.2 s** |
| 1024² footprint | 5.21 GB | **5.32 GB** |
| 1280² 1 ステップ | 48.91 s | **35.21 s** |
| 1280² footprint | 6.89 GB | **6.46 GB** |
| I/O / ブロック | 71.9 ms (6.24 GB/s) | **138 ms (3.34 GB/s)** ← mini の SSD は遅い |
| 計算 / ブロック(1024²) | 976.9 ms | **607 ms** |
| 計算 ÷ I/O | 13.59 | **4.16**(基準 2.0) |
| swapouts | 0 | **0** |
| `iogpu.wired_limit_mb` | 15360(残骸) | **0(既定のまま)** |

送ったもの: リポジトリ 256 MB と `krea2-lowram` 17 GB だけ。**元の q8 スナップショット
21 GB は要らない**(mini 側で bake も量子化もしない)。Thunderbolt bridge 越しの ssh で
約 290 MB/s。手順は M7 §2。

**neural accelerator は効いている。** 生成のホットパスの `mx.quantized_matmul` が
`affine_qmm_t_nax_*` にディスパッチされることを Metal のキャプチャで確認した
(`tools/bench/nax_probe.py`)。合成ベンチでは q8 matmul が M3 Pro の
5.11 → **17.16 TFLOPS**。**q8 が素の bf16 より速いのが neural accelerator の指紋**
(M3 Pro では逆に遅い)。

**同じ機械では出力はピクセル完全一致。機械をまたぐと一致しない**
(平均 4.6 / 255、目視では同じ)。カーネルが違うので q8 matmul の累積順序が変わる。
**クロスマシンのビット一致を約束しないこと。**

## 6. 次にやること

- **M3 Pro での `iogpu.wired_limit_mb=0` 再測定。** M6 の 2 本は前セッションが残した
  15360 の設定下。mini 側は既定 0 で通ったので、残っているのは M3 Pro だけ。
  `sudo sysctl -w iogpu.wired_limit_mb=0` のあと README の §実行のしかた を 1024² で 1 回。
  期待値は 29.6 s/step / footprint 5.2 GB / swapouts 0 / 前回とピクセル一致。
- **DiT の活性を bf16 にする(M6 mini で 1.4 倍以上の見込み)。** bf16 の nax カーネルは
  float32 の 1.68 倍速い。mflux は RoPE / RMSNorm を float32 で通す設計なので
  (`rope_embedder.py`、`common.py`)、ブロック内の matmul も float32 で走っている。
  計算 607 → 360 ms 程度、1 ステップ 21.2 → 14.2 s の見込み。
  **出力の数値は変わるので、M5b と同じく測って画像を見比べて判断する。**
- **前処理ツールの昇格。** `bake_lora_checkpoint.py` と `quantize_te_checkpoint.py` は
  まだ `tools/bench/` にいる。低メモリ用スナップショットを 1 コマンドで作る
  CLI にすれば、シンボリックリンクを手で張る手順が消える。
- **プリフェッチ。** M3 Pro では比 13.59 でやる理由がなかったが、**mini では 4.16**。
  I/O がステップ時間の 19% を占める。bf16 化のあとならもっと効く。

**TE の q8 について**: 「TE の量子化は禁止」と書いてあったのは
**エージェントが根拠なく足した行**で、ユーザの要件ではなかった(`b01d4d7`)。
実測したら常駐が半分になり、画像は目視で同じだった。
[記録](measurements/2026-09-22-m5b-text-encoder-q8.md)

### M6 の実装で踏んだもの(次に触るとき用)

- **`mx.compile` は切る。** ストリーミングされたブロックは forward の中でディスクを
  読み `mx.eval` を呼ぶので、トレースに乗らない。
- **TE を手放す前に `mx.eval(embeds)`。** 遅延評価のままだとグラフがエンコーダの
  ウェイトを掴んだままで、参照を捨てても 1 バイトも解放されない。
- **毎回新しい lazy ハンドルを読む。** 一度読んだツリーを使い回すと評価済みの
  配列がそこから参照され続け、drop しても 1 バイトも解放されない。
- **`mx.eval(out)` をブロックごとに入れる。** 入れないとグラフ未評価のまま
  ウェイトを drop することになる。
- **`staged_loader` はクラス属性の既定値で持つ。** `Krea2.__new__(Krea2)` で組み立てる
  テストがあるので、`__init__` だけで設定すると AttributeError になる。
- **`bake_and_strip_lora` を transformer 全体に呼ばない。**
  `lora_saver.py:125` が層ごとに eval するので全部実体化する(実測 13.10 GB、失敗)。
  ブロック単位で呼ぶ。
- **`nn.quantize` はルートモジュール自身を置換できない。** 部分モジュールを直接渡すと
  その中の子だけが量子化され、渡したモジュール自身は素通りする(`embed_tokens` が
  bf16 のまま残って気づいた)。**評価前にモデル全体を 1 回で量子化する。**
- **TE を q8 で読むには `skip_quantization=False` が要る。** 定義が True のままだと
  量子化構造を作らずに packed な q8 テンソルを update することになる。
- **VAE は DiT を破棄してから構築する。** VAE は `mx.get_peak_memory()` が
  4.40 GB でも実 footprint は(タイル 512 で)14.15 GB ある。
- VAE のタイルは **256** が既定(`--block-streaming` が入れる)。ただし
  **タイルサイズを変えると出力が変わる**ので、再現性のために固定して記録する。
- **swapwatch の swapout カウンタはマシン全体。** 数十ページ規模の値は自分の実行と
  区別できない。1024² の実行で出た 56 ページは、同じサンプルで swap used が 21 MB
  減っていたので外来だった。**より重い 1280² が swapouts 0 で通って決着した。**

## 7. 測定の作法(踏んだ地雷)

- **ベンチのループは内側で `mx.eval`。** 外でまとめて eval すると未使用グラフが
  捨てられ、実際より速い数字が出る。初回 20.9 TFLOPS という誤測定をこれで出した
  (正しくは 5.9)。
- **`ps rss` でメモリを測らない。** §3 の計器の節を読むこと。
- **`footprint -p` の出力を信じない**(整数 GB に丸める)。`proc_pid_rusage` を読む。
- **同じ数字が繰り返し出たら、現象ではなく計器を疑う。** 「3 回とも 1 MB 単位で
  同じ」は物理現象ではなく丸めの症状だった。
- **2 本の計器が食い違ったら、結論を出す前に原因を潰す。**
- **子プロセスの stdout は `PYTHONUNBUFFERED=1` を付ける。** swapwatch に殺されると
  バッファに溜まった出力が消え、どこまで進んだか分からなくなる。
- **index レベルの観察で物理配置を語らない。** 「ブロック順に連続配置」と一度
  結論したが、バイトオフセットを見たら入り組んでいた。
- **見積りで「載らない」と結論しない。** 一方で、載せるために画質を落とす案も出さない。
- 数字は別経路で sanity check する。スクリプトは `tools/bench/` に残し、結果は
  `docs/16gb/measurements/` に日付・機材つきで置く。
- **`just format` をリポジトリ全体にかけない。** ruff 0.16.3 は markdown 内の python
  コードブロックまで整形するので、`docs/16gb/` の既存メモに無関係な差分が出る。
  自分が触ったファイルだけを指定して整形する。

## 8. リポジトリの状態

ブランチ `feat/krea2-block-streaming`。**M6 まで `fork` に push 済み**(`406a0c6`)。
以前の引き継ぎが「`dd87bce` までが push 済み」と書いていたのは古い記録だった。
push は毎回明示の承認が要る(RULE.md)。**`origin` は upstream なので push しない。**

### M6 で upstream のファイルに入れた変更

ここまでは fork 固有のファイルだけで済んでいたが、M6 で本体に入った。

| ファイル | |
|---|---|
| `models/krea2/weights/krea2_weight_stream.py` | 新規。`Krea2BlockStream` / `Krea2StreamedBlock` |
| `models/krea2/krea2_staged_loader.py` | 新規。`Krea2StagedLoader` |
| `models/krea2/krea2_initializer.py` | `block_streaming=True` のとき何も構築しない経路 |
| `models/krea2/variants/txt2img/krea2.py` | `with self._component(...)` 3 か所、`mx.compile` の回避、クラス属性 `staged_loader` |
| `models/krea2/cli/krea2_generate.py` | フラグの受け渡し、タイル 256 の既定 |
| `cli/parser/parsers.py` | `add_block_streaming_arguments()`(krea2 の parser だけが呼ぶ) |
| `tests/test_krea2_block_streaming.py` | 新規。5 件。ストリーミングした forward が常駐版と**ビット一致**することを見る |
| `_typos.toml` | 1 行(`nax` を辞書に追加)。M1 以前から |

**常駐モードの挙動は変えていない。** `_component` は素通りする。
fast テストは 1504 件すべて緑。

### M7 で足したもの

| ファイル | |
|---|---|
| `docs/16gb/measurements/2026-09-22-m7-m6-mac-mini.md` | 新規。M6 mini の実測 |
| `docs/16gb/runs/m6mini/*.csv` | swapwatch のログ 3 本 |
| `docs/16gb/runs/images/20260922-14*-m7-m6mini-*.png` | 1024² / 1280² の出力 |
| `tools/bench/matmul_probe.py` | 新規。dtype 別の matmul / quantized matmul の TFLOPS |
| `tools/bench/nax_probe.py` | 新規。Metal のキャプチャでディスパッチされたカーネル名を読む |
| `tools/bench/qmm_spy.py` | 新規。実際の生成が出す `quantized_matmul` の形を全部記録する |

**`src/` は 1 行も触っていない。** M7 は計測だけ。

### 実行中の一時設定

**M3 Pro の `iogpu.wired_limit_mb` が `15360` のままになっている**(再起動で 0 に戻る)。
**ストリーミングにこの設定は不要**なので 0 に戻し、その状態で測り直すこと(§6)。
**M6 mini は既定の 0 のままで通っているので、そちらは対応不要。**
