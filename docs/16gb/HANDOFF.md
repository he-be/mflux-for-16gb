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
iogpu.wired_limit_mb              14336 MB (14.34 GB)
max_recommended_working_set_size  15.03 GB
max_buffer_length                  9.66 GB
mx.set_wired_limit                既定 0(MLX は何も wire しない)
```

### q8 チェックポイントのサイズ

| | |
|---|---|
| DiT 全体 | 13.62 GB(blocks 12.92 + globals 0.71) |
| 1 ブロック | 461.3 MB(28 ブロックすべて同一、29 テンソル) |
| 4step LoRA | 0.44 GB(bf16, rank 64) |
| text encoder (Qwen3-VL-4B, bf16) | 8.05 GB |
| VAE | 0.51 GB |
| 全部同時 | 22.2 GB → **載らない** |
| **DiT + LoRA のみ** | **14.06 GB** → ここが勝負どころ |

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
  `ps` は 0.03 GB と報告する。`phys_footprint` なら 3.84 GB。
- 判定に使うのは `mx.get_peak_memory()`(プロセス内)と `footprint -p <pid>` の
  `phys_footprint` / `phys_footprint_peak`(外から、ツリー全体を合計)。2 本が一致する
  ことを毎回確認する。
- `swapwatch` はこれに直してある。直す前は `uv run` の launcher を測っていて、8 GB
  常駐の実行を「peak child RSS 0.03 GB」と報告していた。
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

## 4. まだ分かっていないこと(次に測ること)

1. **q8 DiT (14.06GB) をクリーンな状態で常駐させられるか。** アクティベーション量が未測定。
2. **実ブロック 1 個の forward 時間。** これが分からないとストリーミングで I/O が
   隠れるか判断できない。I/O 側は 74 ms/ブロックで確定済み。
   計算 ÷ I/O ≥ 2.0 が合格基準。

## 5. 次の手順

**M1 から。このために再起動が必要で、そこが今の停止点。**
mflux 本体のコードは M6 まで触らない。

### M1. クリーンな状態の測定 ← いまここ

常駐アプリを落として再起動し、ベースラインを記録する:

```sh
uv run python tools/bench/memstat.py --label clean-boot --json docs/16gb/runs/$(date +%Y%m%d-%H%M)-m1-baseline.json
```

参考(汚れた状態での値): claimable 7.5〜11.0 GB、swap 既使用 492 MB、
compressor 1.26 GB。**14.06 GB には足りない。** 再起動後にどこまで広がるかが土俵。

### M2. TE だけのプロセス ← 済。M1 の後に再実行して数字を揃える

```sh
uv run python tools/swapwatch.py --csv docs/16gb/runs/$(date +%Y%m%d-%H%M)-m2-te.csv -- \
  uv run python tools/bench/te_encode.py \
    --model ~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/ad5c0b1784c486bd45c2ced8a9f10aa45293a7c8 \
    --out ~/Library/Caches/mflux/16gb-bench/krea2-embeds.safetensors \
    --json docs/16gb/runs/$(date +%Y%m%d-%H%M)-m2-te.json
```

### M3. DiT だけのプロセス ← 本命。スクリプトは書いてあり、配線は検証済み

```sh
uv run python tools/swapwatch.py --csv docs/16gb/runs/$(date +%Y%m%d-%H%M)-m3-dit.csv -- \
  uv run python tools/bench/dit_steps.py \
    --model ~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/ad5c0b1784c486bd45c2ced8a9f10aa45293a7c8 \
    --embeds ~/Library/Caches/mflux/16gb-bench/krea2-embeds.safetensors \
    --out ~/Library/Caches/mflux/16gb-bench/krea2-latents.safetensors \
    --json docs/16gb/runs/$(date +%Y%m%d-%H%M)-m3-dit.json
```

測るのは `mx.get_peak_memory()` / 1 ステップの実時間 / swapwatch verdict /
`phys_footprint_peak`。無理のかけ方は段階的に(スワップしたら次へ):

1. そのまま(上のコマンド)
2. `--cache-limit-gb 1 --clear-cache-each-step`
3. `sudo sysctl -w iogpu.wired_limit_mb=<もっと大きく>` してから `--wired-limit-gb 14`
4. `--width 768 --height 768`(1024² が駄目だった証拠として記録)

`--no-lora` で LoRA の 0.44 GB を切り分けられる。`--compile` は mflux 本体と同じ
`mx.compile` 経路を試すとき。

### M4. VAE だけのプロセスでデコード → **実画像 1 枚**(スクリプト未作成)

### M5. ブロック単位ストリーミングの実測(M3 の結果にかかわらず)

実ブロック 1 個を bind → forward → drop して、計算 ÷ I/O を測る。
合格基準: 比 ≥ 2.0、drop 後の常駐がベースライン +100MB 以内。

### M6. mflux 本体への実装(数字が出た方式だけ)

### M7. 実運用(2 回実行して画像一致、1280² も)

## 6. 測定の作法(踏んだ地雷)

- **ベンチのループは内側で `mx.eval`。** 外でまとめて eval すると未使用グラフが
  捨てられ、実際より速い数字が出る。初回 20.9 TFLOPS という誤測定をこれで出した
  (正しくは 5.9)。
- **`ps rss` でメモリを測らない。** §3 の計器の節を読むこと。
- **index レベルの観察で物理配置を語らない。** 「ブロック順に連続配置」と一度
  結論したが、バイトオフセットを見たら入り組んでいた。
- **見積りで「載らない」と結論しない。** 一方で、載せるために画質を落とす案も出さない。
- 数字は別経路で sanity check する。スクリプトは `tools/bench/` に残し、結果は
  `docs/16gb/measurements/` に日付・機材つきで置く。
- **`just format` をリポジトリ全体にかけない。** ruff 0.16.3 は markdown 内の python
  コードブロックまで整形するので、`docs/16gb/` の既存メモに無関係な差分が出る。
  自分が触ったファイルだけを指定して整形する。

## 7. リポジトリの状態

ブランチ `feat/krea2-block-streaming`。`dd87bce` までは `fork` に push 済み
(以前の引き継ぎに残っていた「4 コミットがローカルのみ」は解消済み)。
push は毎回明示の承認が要る(RULE.md)。

upstream のファイルで触ったのは `_typos.toml` の 1 行だけ(`nax` を辞書に追加)。
`tools/swapwatch.py` は fork 固有のファイルで、計器の修正で書き換えてある。
