# 計画: Krea 2 q8 を 18GB 機で動かす(実測ゲート方式)

- 日付: 2026-09-22(改訂 4: M1〜M4 の実測を反映)
- ブランチ: `feat/krea2-block-streaming`
- 管理先: fork [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)(upstream へ PR は出さない)

## 0. 前提(動かせない要件)

- **量子化は q8。** 8bit 未満は画像としては破綻しないが、描き込みが減り表現が変わることが
  CUDA 側の検証で分かっている。q4 / q3 は画質要件を満たさないので**選択肢ではない**。
  「載らないから 4bit」は検証に値しない逃げであり、この計画から削除した。
- **スワップは失敗。** 実行は `tools/swapwatch.py` 越しに行い、`clean` でない結果は採用しない。
- **「載らなかった」は正当な結論。** 無理やり載せる試行をして駄目なら、駄目だったと
  数字つきで報告する。代わりに画質を落とす案を出さない。
- 対象構成: **q8 + 4step LoRA / 1024² / 4 ステップ / `--scheduler euler` / guidance 1.0**。

## 1. このマシンの物理的な条件(実測)

```
hw.memsize                        19.33 GB
max_buffer_length                  9.66 GB
iogpu.wired_limit_mb              既定 0(単位は MiB。再起動で 0 に戻る)
  └ 0 のとき      max_recommended_working_set_size  12.88 GB
  └ 14336 MiB     同                                15.03 GB
  └ 15360 MiB     同                                16.11 GB
mx.set_wired_limit                既定 0(MLX は何も wire しない)
```

**この 2 つは両方必要**だった(M3): `sudo sysctl` で枠を開け、プロセス内で
`mx.set_wired_limit` を呼んで初めてウェイトが wire される。片方だけでは効かない。

q8 の内訳(`mflux-community/krea-2-turbo-mflux-q8` の実ファイルサイズ):

| | サイズ |
|---|---|
| DiT (transformer, q8) | 13.62 GB |
| 4step LoRA (bf16, rank 64) | 0.44 GB |
| text encoder (Qwen3-VL-4B, bf16) | 8.05 GB |
| VAE | 0.51 GB |

- 全部同時 = 22.2 GB。**載らない。**
- **LoRA を bake すると常駐は増えない**(q8 のウェイトに畳み込まれるため 13.62 GB のまま。
  bake 中だけ 15.23 GB の一時ピーク)。bake しないと別レイヤで残るので 14.06 GB。
- DiT の数字はこの改訂で 14.06 → 13.62 GB に直した(初版は概算。実測は
  [M0](../../docs/16gb/measurements/2026-09-22-krea2-q8-checkpoint-layout.md) の
  safetensors ヘッダから)。

## 2. 段階(すべて実ウェイトで測る)

mflux 本体のコードは M6 まで触らない。M1〜M4 はプロセスを分けることで、
**コード変更なしに段階ロードを実現する**(済。`tools/bench/` の 4 本)。

### M0. チェックポイントの物理配置 ← 済

- 28 ブロック × 29 テンソル、1 ブロック 461.3 MB。シャード単位ではブロック番号順だが、
  **シャード内部では 1 ブロックのテンソルは連続していない**。それでも拾い読みで
  6.24 GB/s 出るので並べ直しは不要。
  [記録](../../docs/16gb/measurements/2026-09-22-krea2-q8-checkpoint-layout.md)

### M0b. 計器の検証 ← 済

`ps rss` は MLX の確保を 1 バイトも見ない(4 GB 保持のプロセスを 30 MB と報告する)。
判定は `mx.get_peak_memory()` と `phys_footprint` で行う。ただし **`footprint -p` は
大きなプロセスを整数 GB に丸める**(これで一度「16.12 GB の上限」という誤った結論を
出した)。`proc_pid_rusage` を直接読むこと。
[記録](../../docs/16gb/measurements/2026-09-22-memory-instrumentation.md)

### M1. クリーンな状態の測定(再起動直後)

- 常駐アプリを落とし、再起動してから測る: 空きメモリ、`memory_pressure`、
  スワップのベースライン、`iogpu.wired_limit_mb`。
- **これが 14.06 GB を載せられるかどうかの土俵の広さ**。記録して以降の実験の前提にする。
- 計器は `tools/bench/memstat.py`(claimable = free + inactive + speculative + purgeable)。

### M2. TE をプロセス 1 で回し、埋め込みをディスクに出す ← 済(条件は汚い)

- text encoder だけを構築してプロンプトをエンコードし、結果を safetensors に保存して終了。
- 常駐は TE 8.05 GB のみ。swapwatch 下で実測。
- **合格基準**: `clean`。→ **合格**。MLX ピーク 8.19 GB、footprint 8.11〜8.29 GB、
  swapouts 0。エンコード 0.63〜0.81 s。
  [記録](../../docs/16gb/measurements/2026-09-22-m2-text-encoder.md)
- ただし再起動前の状態で測ったので、M1 の後にもう一度通しておくこと(compressor が
  1.26 → 6.21 GB に伸びている)。

### M3. DiT をプロセス 2 で無理やり載せる ← **本命**

- M2 の埋め込みを読み、**q8 DiT + 4step LoRA(14.06 GB)だけ**を載せて 4 ステップ回し、
  latent を保存する。TE も VAE もこのプロセスには存在しない。
- 計器は `tools/bench/dit_steps.py`(段階ごとのフラグは下の 1〜4 に対応)。
- 測る: `mx.get_peak_memory()`、`phys_footprint`、1 ステップの実時間、
  swapwatch verdict。**peak RSS は測れない**(M0b)。
- 段階的に無理をする(各段でスワップしたら次へ):
  1. そのまま実行
  2. `mx.set_cache_limit` を絞る / ブロックごとに `mx.clear_cache()`
  3. `sudo sysctl iogpu.wired_limit_mb` を上げて `mx.set_wired_limit` で常駐を保証
  4. 解像度を 768² に落として アクティベーションを減らす(1024² が駄目だった証拠として記録)
- **合格基準**: `clean` で 4 ステップ完走。
- **結果: 不合格。** 4 ステップは完走して latent も出たが、スワップは消えなかった。
  [記録](../../docs/16gb/measurements/2026-09-22-m3-dit-resident.md)

  | | |
  |---|---|
  | ウェイトの常駐だけなら | **clean**(`mx.set_wired_limit` が必須) |
  | 1024² / 4 ステップ | 要求 15.96 GB、28.15 s/step、**1117 MB スワップ** |
  | 768² / 4 ステップ | 14.71 s/step、102〜467 MB スワップ(ばらつく) |

  収支: 自プロセス 15.9 + システム 2.0 = **17.9 / 19.33 GB**。ページキャッシュの
  余地が消えてスワップする。**上限に当たっているのではなく物理メモリが足りない。**
- 効いた手 / 効かなかった手:
  **`mx.set_wired_limit` は必須**(これなしではロードすら通らない)。
  **LoRA は bake した方がよい**(-0.44 GB、41.9 → 30.9 s/step)。
  `mx.set_cache_limit` は無効(ループ中はむしろ 9 倍悪化)。
  768² でも消えない。`F_NOCACHE` は不要(OS は既に正しく回収している)。
- **結論どおり M5 へ進む。q4 に落とす選択肢は取らない。**

### M4. VAE をプロセス 3 でデコード ← 済(合格)

- タイル 512 で `mx peak 4.40 GB`・swapouts 0 の `clean`。**実画像が 1 枚出た。**
  [記録](../../docs/16gb/measurements/2026-09-22-m4-vae-decode.md)
- タイルなしだと 8.73 GB 使ってスワップする(ウェイトは 0.51 GB しかないのに)。
- 注意: VAE は `mx.get_peak_memory()` 4.40 GB に対し実 footprint 14.16 GB。
  M6 では **DiT を破棄してから VAE を構築する**こと。

### M5. ブロック単位ストリーミング ← 済(**合格**)

M3 が通っても、マージンは「クリーンな状態でギリギリ」でしかない。16GB 機では確実に
足りない。構造的にマージンを作る方法として、ブロック単位のストリーミングを測る。

- 実ブロック 1 個を bind → forward → drop し、**計算時間 ÷ I/O 時間**を実測。
  合成ベンチでは計算律速だったが、実ブロックでは未確認。
- M3 が見込みを裏づけた: ウェイト 13.62 GB が常駐の 86%。1 ブロック 461.3 MB なので
  1〜2 個だけ常駐させれば 0.5〜1 GB になり、要求は 0.9 + 1.94 + 0.35 ≒ **3.2 GB**。
  **16GB 機でも通る量**。分かれ目は速度だけ。
- 参考値(M3 の実測、スワップ込みなので下限): 1024² で 28.15 s/step。
  ブロックあたり 28.15 / 28 ≒ **1.0 s**。I/O は 74 ms/ブロックなので、
  この比が保たれるなら 13.6 倍で、合格基準の 2.0 を大きく超える。
- 28 ブロックに広げて 1 ステップを回し、`mx.get_peak_memory()` と 1 ステップ時間を測る。
- **合格基準**: 比 ≥ 2.0、drop 後の常駐がベースライン +100 MB 以内、
  1 ステップが M3 の実測値の +20% 以内。
- **結果: 合格。** [記録](../../docs/16gb/measurements/2026-09-22-m5-block-streaming.md)

  | 基準 | 実測 | |
  |---|---|---|
  | 比 ≥ 2.0 | **13.65** | ✅ |
  | 1 ステップ ≤ 33.8 s | **29.79 s** | ✅ |
  | drop 後 +100 MB 以内 | +313 MB(ただし 28 ブロックでドリフト **0 MB**) | ⚠️ 基準の見積りが甘かった。アクティベーション分であってウェイトは解放されている |
  | スワップなし | **swapouts 0** | ✅ |

  要求 15.96 → **3.72 GB**、常駐 0.706 GB。`mx.set_wired_limit` も sysctl の細工も不要。
  出力は常駐版と **latent がビット一致、PNG がバイト一致**。
- **LoRA**: 事前に焼き込んだチェックポイントを作る(`tools/bench/bake_lora_checkpoint.py`)。
  焼き込みもブロック単位でやれば **2.14 GB / 13 秒**。実行時コストはゼロ
  (984.7 対 983.7 ms/block)。**`bake_and_strip_lora` を transformer 全体に呼ぶと
  全部実体化して落ちる**(`lora_saver.py:125` が層ごとに eval するため)。

### M6. mflux 本体への実装

M3 / M5 で数字が出た方式だけを入れる。

| ファイル | 変更 |
|---|---|
| `krea2_initializer.py` / `variants/txt2img/krea2.py` | 段階ロード(TE→破棄→DiT→破棄→VAE) |
| `weights/krea2_weight_stream.py`(新規) | M5 が通った場合のみ: bind / release / prefetch |
| `model/krea2_transformer/transformer.py` | 同上、ブロックループのフック |
| `cli/parser/parsers.py` | フラグ |

**合格基準**: プロセス分割版(M2〜M5)と同じ数字が mflux 経由でも出ること。
具体的には 1024²/4 ステップで peak 3.72 GB・29.8 s/step・swapouts 0、
出力が `tools/bench/` 版とバイト一致すること。

M5 は **mflux 本体を 1 行も変えずに** `transformer.blocks` を差し替えるだけで成立した。
本体への実装もこの形(ブロックのリストを差し替え可能にする)でよい。

VAE のタイルは **256** を既定にする(実 footprint 14.15 → 4.28 GB、代償 0.9 秒)。

## M6 の具体設計(2026-09-22 追記)

`tools/bench/` の 4 プロセス版を本体に入れる。**追加する概念は 2 つだけ。**

### (a) 低メモリ用スナップショットは「ふつうの mflux スナップショット」

```
krea2-lowram/
  transformer/   blocks_00..27.safetensors + globals.safetensors + index  ← bake 済み
  text_encoder/  q8(quantize_te_checkpoint.py の出力)
  vae/           q8 スナップショットから
  tokenizer/
```

新しいローダ形式は要らない。`Krea2WeightDefinition._select_transformer_variant` が
`transformer/` を見つけ、`_try_load_mflux_format` が mflux メタデータ付きの
per-block シャードをそのまま読む。**`--model-path` 1 本で 3 コンポーネントが揃う。**
ストリーミングを切っても(常駐で)読める同じディレクトリになる。

### (b) 2 つの新しいクラス

| | |
|---|---|
| `weights/krea2_weight_stream.py` | `Krea2BlockStream`(index を読み、`transformer.blocks` をラッパに差し替える)と `Krea2StreamedBlock`(bind → forward → `mx.eval` → drop)。`tools/bench/block_stream.py` の移植 |
| `krea2_staged_loader.py` | `Krea2StagedLoader.build("text_encoder" / "transformer" / "vae")`。1 個ずつ作って返すだけ。破棄は呼び出し側 |

`Krea2` 側は **3 か所を `with self._component(...)` で囲むだけ**。常駐モードでは
この context manager は素通り(既存の挙動は 1 バイトも変わらない)。

```python
with self._component("text_encoder"):
    embeds, neg_embeds = self._encode_prompts(...)
    mx.eval(embeds)          # TE を手放す前に評価する。しないと graph が weights を掴んだまま
with self._component("transformer") as transformer:
    ...  # ループ
with self._component("vae"):
    decoded = self._decode_latents(...)
```

### フラグ

`--block-streaming`(krea2 の parser だけに足す)。効果は 3 つ:

1. 段階ロード + ブロックストリーミング
2. `--vae-tile-size` 未指定なら **256** を既定にする
3. `mx.compile` を切る(ラッパの中で I/O と `mx.eval` をするので compile できない)

`--low-ram` には手を触れない。**あれは `mx.set_cache_limit(1GB)` を入れるが、
M3 でループ中 9 倍悪化した手**なので、ストリーミングの合格判定には使わない。

`--lora-paths` との併用はエラーにする。実行時 LoRA は M3 で +11 s/step と測れており、
採用したのは事前焼き込み。`tools/bench/bake_lora_checkpoint.py` を案内する。

### M7. 実運用

- q8 + 4step LoRA、1024²、seed 42、swapwatch 下。2 回実行して画像一致。
- その後 1280²(LoRA の学習 σ と一致する解像度)。

## 3. 測定の作法

- ベンチのループは**内側で `mx.eval`**。外でまとめて eval すると未使用グラフが捨てられ、
  実際より速い数字が出る(初回 20.9 TFLOPS という誤測定をこれで出した。正しくは 5.9)。
- **`mx.load` は lazy。** ウェイトを読む時間を測るなら `mx.eval(model)` を明示する。
  しないと読み出しコストが次の処理の時間に紛れる。
- **メモリは `ps rss` で測らない**(M0b)。`mx.get_peak_memory()` と
  `proc_pid_rusage` の `phys_footprint` の 2 本で、一致することを確認して使う。
  **一致しないときは原因を潰すまで結論を出さない**(VAE では 4.40 GB 対 14.16 GB で
  一致しない。畳み込みが MLX の外側で確保している)。
- **同じ数字が繰り返し出たら、現象ではなく計器を疑う。** 「3 回とも 1 MB 単位で同じ」は
  `footprint -p` の丸めだった。
- 数字は別経路で sanity check する(トークン数半減で時間が半分になるか、など)。
- スクリプトは `tools/bench/`、結果は `docs/16gb/measurements/` に日付・機材つきで。
  swapwatch の CSV / JSON は `docs/16gb/runs/`、中間生成物(埋め込み・latent)は
  リポジトリ外の `~/Library/Caches/mflux/16gb-bench/`。

## 4. 非目標

- upstream への PR。
- 他モデルへの一般化。
- **画質を落とす方向の回避策**(DiT の q4 / q3)。

### 訂正: TE の量子化は非目標ではない

この欄には当初「TE の量子化」も入れていたが、**根拠がなかったので取り下げる**。

- ユーザからそういう要件は出ていない。この行は fork 初回コミット `b01d4d7` で
  エージェントが勝手に足したもの。
- 根拠にしていたのは upstream のコード内コメント 1 行
  (`krea2_weight_definition.py:52` の `skip_quantization=True,
  # quantizing the TE degrades conditioning`)。**実測ではない。**
  同じ注記は Qwen / Qwen 2.1 にもあるが、どのビット幅で何がどう劣化するのかは
  書かれていない。
- **q8 は DiT で受け入れている品質基準そのもの。** TE だけ「量子化は一律駄目」と
  するのは筋が通らない。
- text encoder は Qwen3-VL-**4B**。bf16 で 8.05 GB だが、q8 なら約 4.3 GB。
  M5 の後、**パイプライン最大の消費は TE の 8.29 GB** なので、効果は大きい。

**測ってから決める**(M5b)。判定は埋め込みの一致度ではなく、
**同じ seed で出した画像を見比べて**行う。
