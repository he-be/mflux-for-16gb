# 引き継ぎ(2026-09-22 時点)

新しいセッションはこのファイルだけ読めば再開できる。次に読むのは
[計画](../../.cursor/plans/2026-09-22-krea2-block-streaming.md)。

## 1. やろうとしていること

M3 Pro 18GB の MacBook Pro で **Krea 2 Turbo q8 + 4step 蒸留 LoRA** の画像生成を、
**スワップさせずに**通す。その後 M6 mac mini でも同じ構成を回す。

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
| fork remote | `fork` = `https://github.com/he-be/mflux-for-16gb`(**PR は出さない**) |
| ブランチ | `feat/krea2-block-streaming` |
| q8 ウェイト | `~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/ad5c0b1784c486bd45c2ced8a9f10aa45293a7c8/` |
| 4step LoRA | `~/Library/Caches/mflux/loras/krea2_turbo_4step_rank_64_lora_comfyui.safetensors` |
| 中間生成物 | `~/Library/Caches/mflux/16gb-bench/`(埋め込み・latent・**q8 TE 4.3GB** と**焼き込み済み DiT 13GB**。リポジトリ外) |
| スワップ見張り | `tools/swapwatch.py` |
| 計測スクリプト | `tools/bench/`(`memstat.py` / `te_encode.py` / `dit_steps.py` / `vae_decode.py` / `block_stream.py` / `bake_lora_checkpoint.py` / `quantize_te_checkpoint.py`) |
| 文書 | `docs/16gb/`(research / measurements / runs)、計画は `.cursor/plans/` |

ダウンロードは完了済み(q8 21GB + LoRA 438MB)。再取得は不要。LoRA は
`lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors`
という指定でも通る(mflux が自前のキャッシュに写しを持っている)。

## 3. 確定した事実(すべてこのマシンでの実測)

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
- `Krea2Initializer.init` は 3 コンポーネントを構築して `mx.eval(model)` で一括実体化する
  (`krea2_initializer.py:30-38`)。段階ロードは未実装。**`tools/bench/` はこれを迂回して
  コンポーネント単位で `WeightLoader.load_single_local` を呼ぶ。**
- q8 DiT の配線は検証済み: 956 パラメータが 956/956 で一致、形の不一致 0、
  量子化層 263、`bits=8`。
- **`mx.load` は lazy。** 上の検証は 13.62 GB を一切実体化せずに通る(active 0.000 GB)。
  読み出し時間を測るなら `mx.eval(model)` を明示すること。
- `MemorySaver` はエンコード後に TE を、`--low-ram` はループ後に DiT を破棄する
  (`memory_saver.py:77`)。

## 4. 結論: **目標構成は 18GB 機でスワップなしに通る**

方法は**ブロック単位ストリーミング**(M5)。常駐方式(M3)は不可能だった。

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
出力は `docs/16gb/runs/images/`。**常駐版とバイト単位で一致**している。

### 各段の結果

| 段 | 結果 |
|---|---|
| M1 ベースライン | 済。`iogpu.wired_limit_mb` の既定が 0 と判明 |
| M2 TE | **合格 clean** |
| M3 DiT 常駐 | **不合格**。要求 15.96 GB。4 ステップは完走するがスワップする |
| M4 VAE | **合格 clean**。タイル 256 |
| M5 ストリーミング | **合格 clean**。要求 3.72 GB、比 13.65、ビット一致 |
| M5b TE を q8 に | **採用**。8.29 → 4.96 GB。画像は目視で同じ、描き込みは同等 |

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

## 5. 次にやること

### M6: mflux 本体への実装 ← いまここ

いま動くのは `tools/bench/` の分割プロセス版だけ。本体に入れる。

| ファイル | 変更 |
|---|---|
| `krea2_initializer.py` / `variants/txt2img/krea2.py` | 段階ロード(TE→破棄→DiT→破棄→VAE) |
| `weights/krea2_weight_stream.py`(新規) | ブロックの bind / drop。`tools/bench/block_stream.py` がそのまま雛形になる |
| `model/krea2_transformer/transformer.py` | ブロックのリストを差し替え可能にする(今回はテスト側から差し替えた) |
| `cli/parser/parsers.py` | フラグ |

**合格基準**: 1024²/4 ステップで peak 3.72 GB・29.8 s/step・swapouts 0、
出力が `tools/bench/` 版とバイト一致すること。

**実装上の注意(実測で踏んだもの)**

- **毎回新しい lazy ハンドルを読むこと。** 一度読んだツリーを使い回すと評価済みの
  配列がそこから参照され続け、drop しても 1 バイトも解放されない。
- **`mx.eval(out)` をブロックごとに入れること。** MLX は遅延評価なので、
  入れないとグラフ未評価のままウェイトを drop することになる。
- **`bake_and_strip_lora` を transformer 全体に呼ばないこと。**
  `lora_saver.py:125` が層ごとに eval するので全部実体化する(実測 13.10 GB、失敗)。
  ブロック単位で呼ぶ。
- **`nn.quantize` はルートモジュール自身を置換できない。** 部分モジュールを直接渡すと
  その中の子だけが量子化され、渡したモジュール自身は素通りする(`embed_tokens` が
  bf16 のまま残って気づいた)。**評価前にモデル全体を 1 回で量子化する。**
- **TE を q8 で読むには `skip_quantization=False` が要る。** 定義が True のままだと
  量子化構造を作らずに packed な q8 テンソルを update することになる。
  `TextEncoderQuantizer.loadable_component()` がこれをやっている。
- **VAE は DiT を破棄してから構築すること。** VAE は `mx.get_peak_memory()` が
  4.40 GB でも実 footprint は(タイル 512 で)14.15 GB ある。
- VAE のタイルは **256** を既定に。ただし**タイルサイズを変えると出力が変わる**ので、
  再現性のために固定して記録する。

### M7: 実運用

2 回実行して画像一致、1280²(LoRA の学習 σ と一致する解像度)。

### その先

**どの段も 5 GB を超えていないので、16GB 機には十分な余裕がある。**
次の削りどころを探すより、M6 / M7 を通して実機で確認する方が先。

**TE の q8 について**: 「TE の量子化は禁止」と書いてあったのは
**エージェントが根拠なく足した行**で、ユーザの要件ではなかった(`b01d4d7`)。
実測したら常駐が半分になり、画像は目視で同じだった。
[記録](measurements/2026-09-22-m5b-text-encoder-q8.md)

## 6. 測定の作法(踏んだ地雷)

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

## 7. リポジトリの状態

ブランチ `feat/krea2-block-streaming`。**`dd87bce` までが `fork` に push 済みで、
それ以降の M1〜M4 のコミットはすべてローカルのみ。**
push は毎回明示の承認が要る(RULE.md)。

upstream のファイルで触ったのは `_typos.toml` の 1 行だけ(`nax` を辞書に追加)。
`tools/swapwatch.py` と `tools/bench/` は fork 固有。

### 実行中の一時設定

`sudo sysctl -w iogpu.wired_limit_mb=15360` が入ったままになっている可能性がある
(再起動で 0 に戻る)。**ストリーミング方式にはこの設定は不要**なので、
戻してよい。確認は `sysctl -n iogpu.wired_limit_mb`、戻すのは
`sudo sysctl -w iogpu.wired_limit_mb=0`。

**むしろ 0 に戻した状態で M6 の合格判定をすること。** 特別な設定なしで通ることが
この方式の価値なので。
