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
| 中間生成物 | `~/Library/Caches/mflux/16gb-bench/`(M2 の埋め込みなど。リポジトリ外) |
| スワップ見張り | `tools/swapwatch.py` |
| 計測スクリプト | `tools/bench/`(`memstat.py` / `te_encode.py` / `dit_steps.py`) |
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

### M2 の結果(合格、ただし条件は汚い)

TE だけのプロセス: MLX ピーク **8.19 GB**、footprint ピーク 8.11〜8.29 GB、
**swapouts 0**、verdict `clean`。実体化 1.3 s (5.9 GB/s)、エンコード 0.63〜0.81 s。
埋め込みは `(1, 30, 30720)` bf16 = 1.84 MB。
[記録](measurements/2026-09-22-m2-text-encoder.md)

**再起動前の状態で測った**ので、M1 の後にもう一度通しておくこと。

### mflux 側の作法(コードを読んで確認済み)

- `--steps 4` を明示(既定 8)、`--scheduler euler`(既定 er_sde)、guidance は 1.0 のまま
  (mflux の 1.0 = 単一パス = ComfyUI の cfg 1.0)。
- LoRA の全 456 キーが `Krea2LoRAMapping` に 456/456 で一致(検証済み)。
- q8 なら LoRA は bake してよい(8bit 未満のときだけ `--no-bake-lora` が要る)。
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

## 4. 今日やったこと(M1〜M4)と、その結論

### M1〜M4 はすべて実施済み。結論: **常駐方式は 18GB 機では成立しない。**

| 段 | 内容 | 結果 |
|---|---|---|
| M1 | クリーンな状態のベースライン | 済。`iogpu.wired_limit_mb` の既定が 0 だと判明 |
| M2 | TE だけのプロセス | **合格 clean**。8.19 GB、swapouts 0 |
| M3 | **q8 DiT だけのプロセス** | **不合格**。4 ステップは完走するがスワップする |
| M4 | VAE だけのプロセス | **合格 clean**。タイル 512 で 4.40 GB。**実画像が出た** |

詳細はそれぞれ `docs/16gb/measurements/2026-09-22-m{1,2,3,4}-*.md`。

### M3 の中身(一番重要)

| 構成 | 常駐 | 要求 | 1 ステップ | swapouts | 判定 |
|---|---|---|---|---|---|
| ウェイトのみ + wire | 13.62 GB | 13.62 GB | — | **0** | **clean** |
| + LoRA(bake なし)ロードのみ | 14.06 GB | 14.06 GB | — | **0** | **clean** |
| **1024² / 4 ステップ** | 13.63 GB | **15.96 GB** | 28.15 s | 1117 MB | SWAPPED |
| 768² / 4 ステップ | 13.63 GB | 15.2 GB | 14.71 s | 102〜467 MB | SWAPPED |

収支: 自プロセス 15.9 + システム 2.0 = **17.9 / 19.33 GB**。
ページキャッシュが 0.15 GB まで削られ、それでも足りずに毎秒数百ページ掃き出す。
**上限に当たっているのではなく、物理メモリが足りない。**

### 効いた手・効かなかった手(全部実測済み。もう一度試す必要はない)

| 手 | 結果 |
|---|---|
| **`mx.set_wired_limit`** | **必須。これなしではロードすら通らない**(1.4〜3.2 GB スワップ) |
| **`sudo sysctl iogpu.wired_limit_mb`** | 上の前提条件。両方ないと wire されない |
| **LoRA を bake する** | **した方がよい。** 常駐 -0.44 GB、1 ステップ 41.9 → 30.9 s |
| VAE のタイル (512) | 必須。8.73 → 4.40 GB |
| `mx.set_cache_limit` を絞る | **効かない。**ロード中は無関係、ループ中は 9 倍悪化 |
| 解像度 768² | 速度は倍になるがスワップは消えない |
| `F_NOCACHE` でページキャッシュ迂回 | **不要。**OS は既に正しく回収している(仮説は外れ) |

### パイプラインは端から端まで通った

```
M2: TE 8.05 GB   → 埋め込み (1, 30, 30720)
M3: DiT 13.62 GB → latent [1, 16, 128, 128]
M4: VAE 0.51 GB  → 1024×1024 PNG ← docs/16gb/runs/images/
```

画像は意図どおりで、**q8 + 4step LoRA が正しく動くことを確認済み**。

## 5. 次にやること: M5(ブロック単位ストリーミング)

M3 が不合格だったので、計画どおり M5 へ進む。**M3 の数字が M5 の見込みを裏づけている:**

- ウェイト 13.62 GB が要求 15.96 GB の **86%**。
- 1 ブロック 461.3 MB。28 個のうち 1〜2 個だけ常駐させれば **0.5〜1 GB**。
- 要求は 0.9 + 1.94(アクティベーション)+ 0.35 ≒ **3.2 GB**。
  **16GB 機でも桁で余る。**

分かれ目は速度だけ:

| | |
|---|---|
| I/O(確定済み) | **74 ms / ブロック** |
| 計算(未測定) | M3 から逆算すると 28.15 / 28 ≒ **1.0 s / ブロック** |
| 比 | 13.6(合格基準は 2.0) |

ただし 28.15 s/step は**スワップ込みの数字なので下限**であり、実ブロック 1 個の
forward を直接測る必要がある。これが M5 の中身。

### M5 の作り方

`tools/bench/` に 5 本目を書く。実ブロック 1 個を bind → forward → drop して
計算時間と I/O 時間を測り、28 ブロックに広げて 1 ステップを回す。
既存の `dit_steps.py` が `WeightLoader.load_single_local` でコンポーネント単位に
読む形になっているので、そこからブロック単位に降りる。

**合格基準**(計画より): 比 ≥ 2.0、drop 後の常駐がベースライン +100 MB 以内、
1 ステップが 28.15 s の +20% 以内。

### その先

- **M6**: 数字が出た方式を mflux 本体へ。段階ロード(TE→破棄→DiT→破棄→VAE)は
  M2〜M4 で成立が確認できているので、これは入れてよい。
  **VAE は `mx.get_peak_memory()` 4.40 GB に対し実 14.16 GB なので、
  DiT を破棄してから構築すること。**
- **M7**: 実運用(2 回実行して画像一致、1280² も)。

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
(再起動で 0 に戻る)。M5 はストリーミングで常駐を 1 GB 級に落とす話なので、
**この設定は不要**。確認は `sysctl -n iogpu.wired_limit_mb`、戻すのは
`sudo sysctl -w iogpu.wired_limit_mb=0`。
