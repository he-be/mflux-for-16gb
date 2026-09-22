# mflux-for-16gb

16GB / 18GB の Apple Silicon で mflux の大きいモデル(まずは Krea 2)を
**スワップさせずに**動かすための fork。

- upstream: [`mflux-community/mflux`](https://github.com/mflux-community/mflux)
- この fork: [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)
- **upstream へ PR は出さない。** 作業はこの fork で完結させる。

## 方針

1. **量子化は q8 固定。** 8bit 未満は画像としては破綻しないが、描き込みが減り
   表現が変わることが CUDA 側の検証で分かっている。**q4 / q3 に落とす回避策は
   検証対象にしない。** 「載らないなら載らない」と数字つきで結論する。
   **ただし q8 は禁止ではなく基準。** text encoder を q8 にする案は
   「量子化だから駄目」ではなく、**測って画像を見比べて**判断する(M5b)。
2. **計測してから決める。** 見積りだけで結論しない。数字は
   `docs/16gb/measurements/` に日付つきで残す。
3. **スワップは失敗とみなす。** 実行は必ず `tools/swapwatch.py` 越しに行い、
   判定が `clean` でない実行結果は採用しない。
4. **upstream のファイルはできるだけ触らない。** 取り込み時の衝突を減らすため、
   fork 固有の文書は `docs/16gb/` に、計画は `.cursor/plans/` に置く
   (後者は upstream の `.cursor/rules/RULE.md` の規約に従っている)。

## 置き場所

| 場所 | 中身 |
|---|---|
| `docs/16gb/research/` | 調査メモ(ウェイトの入手元、実装の読み解き、モデル固有の作法) |
| `docs/16gb/measurements/` | 実測値。機材・日付・再現コードつき |
| `docs/16gb/runs/` | 実行ログ(swapwatch の CSV / JSON)と `images/` に出力画像 |
| `.cursor/plans/` | 実装計画(RULE.md の規約) |
| `tools/` | fork 固有のツール(`swapwatch.py`) |
| `tools/studio/` | ブラウザ UI(`server.py` / `index.html`)。標準ライブラリのみ |
| `tools/bench/` | 段階ごとの計測スクリプト(`memstat.py` / `te_encode.py` / `dit_steps.py` / `vae_decode.py` / `block_stream.py` / `bake_lora_checkpoint.py` / `quantize_te_checkpoint.py` / `matmul_probe.py` / `nax_probe.py` / `qmm_spy.py` / `ceiling_probe.py` / `mps_probe.py` / `block_budget.py` / `stream_ab.py` / `seqpatch.py`) |
| `~/Library/Caches/mflux/16gb-bench/` | 段の間で受け渡す中間生成物(埋め込み・latent)。リポジトリには入れない |

## 索引

- **[引き継ぎ(2026-09-22)](HANDOFF.md)** ← 新しいセッションはここから
- **[studio — ブラウザから使う](STUDIO.md)** ← 日常の生成はここから

- [Krea 2 を 16GB / 18GB で動かすための調査](research/krea2-low-memory-mac.md)
- [Krea 2 Turbo 4-step 蒸留 LoRA の使い方](research/krea2-4step-lora.md)
- [M3 Pro 18GB: 計算律速と SSD ストリーミングの実測](measurements/2026-09-22-m3pro-compute-vs-ssd.md)（合成ベンチ。実ウェイトでは未測定）
- [M0: Krea 2 Turbo q8 チェックポイントの物理配置](measurements/2026-09-22-krea2-q8-checkpoint-layout.md)
- [計測の土台: macOS で MLX のメモリを何で測るか](measurements/2026-09-22-memory-instrumentation.md)
- [M1: クリーンな状態のベースライン](measurements/2026-09-22-m1-clean-baseline.md)
- [M2: text encoder だけのプロセスで埋め込みを作る](measurements/2026-09-22-m2-text-encoder.md)
- **[M3: q8 DiT を常駐させられるか(結論: 不合格)](measurements/2026-09-22-m3-dit-resident.md)**
- [M4: VAE でデコード → 実画像 1 枚](measurements/2026-09-22-m4-vae-decode.md)
- **[M5: ブロック単位ストリーミング(結論: 合格)](measurements/2026-09-22-m5-block-streaming.md)**
- [M5b: text encoder を q8 にする(結論: 採用)](measurements/2026-09-22-m5b-text-encoder-q8.md)
- **[M6: mflux 本体への実装(結論: 合格。1280² も clean)](measurements/2026-09-22-m6-mflux-cli.md)**
- **[M7: M6 mac mini 16GB の実機(結論: 合格。neural accelerator も効いている)](measurements/2026-09-22-m7-m6-mac-mini.md)**
- **[M8: DiT 1 ブロックの時間の内訳(結論: 遅さの正体は活性の float32)](measurements/2026-09-22-m8-block-profile.md)**
- **[M8a: DiT の活性を bf16 にする(結論: 合格。21.2 → 13.1 s/step、画像は同等)](measurements/2026-09-22-m8a-bf16-activations.md)**
- **[M8b: 次のブロックを計算の裏で読む(結論: 合格。13.1 → 9.4 s/step、出力は完全一致)](measurements/2026-09-22-m8b-prefetch.md)**
- **[M8c / M8d: K 分割(不採用)と clear_cache の廃止(採用)。最終 9.26 s/step](measurements/2026-09-22-m8c-m8d-kslices-cache-limit.md)**
- [M8f: M3 Pro での回帰(合格。29.6 → 23.0 s/step、footprint 4.66 GB)](measurements/2026-09-22-m8f-m3pro.md)
- **[M9: この GPU の天井と MLX が届いていない場所(結論: 天井 19 TFLOPS、`mlp.down` の崖は MLX のカーネル。q4 も MPS も答えではない)](measurements/2026-09-22-m9-ceiling-probe.md)**
- [M9a/b: 本番の再現と 1 ブロックの予算(matmul 256 + sdpa 27 + 要素演算 34。compile は遅くなる、norm の bf16 化だけ −10 ms)](measurements/2026-09-22-m9b-block-budget.md)
- **[M9c: `async_eval` が呼び出し側を止める。直接読み + K4 バッチ + bf16 norm で 9.01 → 7.90 s/step](measurements/2026-09-22-m9c-prefetch-interference.md)**
- **[M10: ストリーミングに実行時 LoRA(結論: 合格。焼き込み不要で差し替え自由、+31%/step・+0.44GB・clean)](measurements/2026-09-22-m10-runtime-lora.md)**
- [計画: ブロック単位ウェイトストリーミング](../../.cursor/plans/2026-09-22-krea2-block-streaming.md)
- [計画: DiT を M6 mini で 2〜3 倍速くする](../../.cursor/plans/2026-09-22-krea2-dit-speed.md)(M8a〜f 済)
- **[計画: DiT を天井(6 s/step)に近づける — 測定と改善](../../.cursor/plans/2026-09-22-krea2-dit-ceiling.md)**(M9a〜e 済、M9f/g 残)

## いま分かっていること(2026-09-22)

**実測済み(事実)**

- このマシン: `hw.memsize` 19.33GB、GPU の max buffer length 9.66GB。
  **`iogpu.wired_limit_mb` の既定は 0**(以前 14336 と記録していたのは手動設定の残骸)。
  既定 0 では `max_recommended_working_set_size` は 12.88GB、14336 MiB で 15.03GB、
  15360 MiB で 16.11GB。**単位は MiB**(14336 MiB = 15,032,385,536 B と一致)。
- q8 の実サイズ: DiT 13.62GB(1 ブロック 461.3MB × 28 + globals 0.71GB)、
  TE 8.05GB、VAE 0.51GB。**4step LoRA は bake すると常駐を増やさない**
  (q8 のウェイトに畳み込まれる)。bake 中だけ一時的に 15.23GB まで上がる。
- 実ウェイトの読み出し: 1 ブロック 461.3MB を **74 ms (6.24 GB/s)**、
  transformer 全体 13.62GB を 3.05 s (4.47 GB/s)。散らばった配置でも帯域は落ちない。
- M3 Pro の行列積は 5.1〜5.9 TFLOPS(合成テンソル)。
- 合成 safetensors 28 個 × 138MB を毎周ディスクから流し直しても、常駐 0.17GB で
  計算律速のまま(bind→eval→drop でメモリは戻る)。
- **`ps rss` は MLX の確保を見ない。** 4GB 保持のプロセスを 30MB と報告する。
  判定は `mx.get_peak_memory()` と `proc_pid_rusage` の `phys_footprint` で行う
  (`footprint -p` は大きなプロセスを整数 GB に丸めるので使わない)。
- **M2 合格**: TE だけのプロセスは 8.05GB 常駐・swapouts 0 で通る。エンコード 0.63〜0.81 s。
- **M3 不合格**: q8 DiT 13.62GB は `mx.set_wired_limit` を使えば**常駐だけはできる**
  (swapouts 0)。しかし 1024² のアクティベーション 1.94GB が乗ると要求は 15.96GB になり、
  システムの 2.0GB と合わせて 17.9 / 19.33GB。ページキャッシュの余地が消えてスワップする。
  768² に落としても消えない。**18GB 機で q8 を常駐方式で回すのは無理。**
- **M4 合格**: VAE はタイル 512 で 4.40GB・swapouts 0。**実画像が 1 枚出た。**
  3 コンポーネントを別プロセスに分ければ、各段は単独で 18GB 機に載る。
- **M5 合格 — これで目標構成が通った。** 28 ブロックをディスクからストリーミングすると
  要求は 15.96 → **3.72GB**、swapouts **0**、1 ステップ 29.8 s(常駐版 28.2 s の +5.8%)。
  計算 ÷ I/O = **13.65**(合格基準 2.0)。出力は常駐版と**バイト単位で一致**。
- **LoRA は事前にチェックポイントへ焼き込む。** 焼き込み自体もブロック単位でやれば
  **2.14GB / 13 秒**で済み、以後の実行時コストはゼロ。

## 目標構成は 18GB 機で通る(2026-09-22 時点、すべて `clean`)

| 段 | mx peak | 実 footprint | 時間 |
|---|---|---|---|
| text encoder(q8) | 4.47GB | **4.96GB** ← 最大 | 2 s |
| LoRA 焼き込み(1 回だけ) | 1.89GB | 2.14GB | 13 s |
| **DiT ストリーミング** | 3.21GB | 3.72GB | **119 s** |
| VAE(タイル 256) | 2.98GB | 4.28GB | 5.6 s |

q8 + 4step LoRA / 1024² / 4 ステップ / euler / guidance 1.0 / seed 42。
画像: `docs/16gb/runs/images/`。

**どの段も 5GB を超えない。** text encoder を q8 にして 8.29 → 4.96GB にした(M5b)。
1 回だけ必要な前処理: TE の量子化(1.19GB / 2.3s)と LoRA の焼き込み(1.89GB / 13s)。

mflux 本体経由(`--block-streaming`)でも同じ: 1024² で footprint 5.21GB / 29.60 s/step、
1280² で 6.89GB / 48.91 s/step、どちらも swapouts 0(M6)。**最新(M9c)は M6 mini で 1024² 7.90 s/step /
6.18 GB、1280² 14.36 s/step / 6.53 GB、M3 Pro で 1024² 22.89 s/step / 5.04 GB。**

- **M6 合格 — mflux 本体に入った。** `--block-streaming` 1 本で、`tools/bench/` の
  4 プロセス分割版と同じ数字が出る(29.60 s/step、計算 ÷ I/O 13.59、I/O 71.9 ms)。
  **出力はピクセル完全一致**(1048576 中 0 ピクセル違い)。
- **1280² も clean。** LoRA の学習 σ と一致する解像度(μ=1.15)で
  48.91 s/step、footprint 6.89GB、**swapouts 0**。1024² より明らかに描き込みが多い。

- **M7 合格 — 本来の目的だった 16GB 機で通った。** Mac mini (Apple M6) 16GB / macOS 27 で
  1024² が 20.9 s/step・footprint 5.32GB、1280² が 35.21 s/step・footprint 6.46GB、
  どちらも **swapouts 0**、`iogpu.wired_limit_mb` は**既定の 0 のまま**。
  M3 Pro より 1 ステップ 1.42 倍速い。**M6 の GPU neural accelerator は効いている**
  (生成のホットパスが `affine_qmm_t_nax_*` に落ちることを Metal のキャプチャで確認)。
  一方 mini の SSD は半分の速さ(3.34 GB/s)で、計算 ÷ I/O は 13.59 → 4.16。

**まだやっていないこと**

- **M3 Pro での `iogpu.wired_limit_mb=0` 再測定**(M6 の 2 本は前セッションが残した
  15360 の設定下。mini 側は既定 0 で通ったので、残っているのは M3 Pro だけ)。
- **M8 済 — DiT は M6 mini で 2.3 倍、M3 Pro で 1.3 倍速くなった。** 遅さの正体は
  ノイズの float32 が DiT 全体に伝播していたこと(M8)。ストリーミング経路で bf16 に落とし
  (M8a)、次のブロックを計算の裏で読み(M8b)、ブロックごとの `clear_cache` を
  cache limit に置き換えた(M8d)。1024² が 21.2 → **9.26 s/step**、1280² が
  35.2 → **15.88 s/step**、M3 Pro が 29.6 → **22.96 s/step**、すべて clean。
  画像は機材差より小さい差。`mlp.down` の K 分割(M8c)は単体では速いが本番では
  プリフェッチと干渉して遅くなるので不採用。
- **M9 済 — 天井を測った。** 密行列積の天井は MLX でも PyTorch MPS でも **19 TFLOPS**(M6 12 コアの実力)。
  1 ステップ 112 TFLOP なので下限は約 6.2 s/step、いまの 9.26 s はその 1.5 倍。差の最大項は
  `mlp.down`(16384→6144)の MLX カーネルが **MPS の半分の速さ**(9.1 vs 17.9 TFLOPS、0.32.2 でも同じ)
  で走っていること(1.3 s/step)。q4 は崖の外で +7% しかなく、MPS(ComfyUI)は逆量子化を払って
  同額かつ 16GB に載らない。**q8 のまま MLX で崖を消す**のが答え。
- **M9c 済 — 1024² が 9.01 → 7.90 s/step、1280² が 15.88 → 14.36、どちらも clean。** M8c で K 分割が
  本番で負けた理由は **`mx.async_eval` が呼び出し側を 100〜250 ms 止める**ことで、主スレッドがその間に
  ディスクを読むと GPU に仕事が届かなくなる。直したのは 3 つ: 次ブロックを **`async_eval` の前に起動した
  スレッドが、先行確保した 2 組の MLX バッファへ `preadv` で直接読む**、`mlp.down` を **K4 のバッチ qmm 1 本**
  にする(92 → 51 ms)、RMSNorm の float32 往復をやめる(−10 ms)。後の 2 つは `--block-streaming` だけが
  有効にする。footprint は 2 組のバッファ分 +0.9 GB(1024² 6.18 GB、1280² 6.53 GB)。画像は目視で同一、
  ピクセル差の平均 2.4〜2.6(M8 の bf16 化 4.7 より小さい)。M3 Pro は 22.89 s/step で ±0。
- **残り**: `powermetrics`(要 sudo)、K=16384 の件を MLX に報告(再現例は `ceiling_probe.py --only ksweep`)、
  M3 Pro を常駐アプリなしで再測定(今回は swapouts 8 ページで clean でない)+ `iogpu.wired_limit_mb=0`、前処理ツールの昇格。

## 実行のしかた

M6 で mflux 本体に入ったので、**フラグ 1 本で通る**(`--block-streaming`)。

### 1 回だけの前処理

`--block-streaming` は「ブロックごとに 1 ファイル」のチェックポイントを要求する。
LoRA も事前に焼き込む(実行時に当てると 1 ステップ +11 秒。M3 の実測)。

```sh
B=~/Library/Caches/mflux/16gb-bench
Q8=~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/<rev>

# DiT: 4step LoRA を焼き込んでブロック単位に書き出す(2.14GB / 13 秒)
uv run python tools/bench/bake_lora_checkpoint.py --model $Q8 --out $B/krea2-q8-4step-baked

# text encoder: 1 回だけ q8 にする(1.19GB / 2.3 秒)
uv run python tools/bench/quantize_te_checkpoint.py --model $Q8 --out $B/krea2-te-q8

# 3 つを 1 つのスナップショットに並べる(新しい形式ではない。ふつうの mflux 配置)
mkdir -p $B/krea2-lowram
ln -sfn $B/krea2-q8-4step-baked    $B/krea2-lowram/transformer
ln -sfn $B/krea2-te-q8/text_encoder $B/krea2-lowram/text_encoder
ln -sfn $Q8/vae                     $B/krea2-lowram/vae
ln -sfn $Q8/tokenizer               $B/krea2-lowram/tokenizer
```

### 生成

```sh
uv run python tools/swapwatch.py --csv docs/16gb/runs/$(date +%Y%m%d-%H%M)-q8-4step.csv -- \
  uv run mflux-generate-krea2 \
    --model ~/Library/Caches/mflux/16gb-bench/krea2-lowram --base-model krea-2 \
    --block-streaming \
    --prompt "..." --seed 42 --steps 4 --scheduler euler --guidance 1.0 \
    --width 1024 --height 1024 --output image.png
```

`--block-streaming` がやること:

1. **段階ロード。** text encoder → 破棄 → DiT → 破棄 → VAE。3 つ同時は 22.2GB で載らない
2. **ブロック単位ストリーミング。** 28 ブロックをディスクに置いたまま bind → forward → drop
3. **VAE タイル 256 を既定にする**(`--vae-tile-size` を明示すればそちらが勝つ)

`--lora-paths` との併用はエラーになる(事前焼き込みを使うこと)。
`--low-ram` は**使わない**。あれが入れる `mx.set_cache_limit(1GB)` は M3 でループを
9 倍悪化させた手で、ストリーミングには不要。

### 別のマシンで回す

M6 mac mini (16GB) への移し方と実測は
**[M7](measurements/2026-09-22-m7-m6-mac-mini.md)** にある。送るのは
リポジトリと**前処理済みの `krea2-lowram` 17GB だけ**で、元の q8 スナップショット
21GB は要らない(bake も量子化もしない)。`rsync -aL` で symlink を実体化する。

`swapwatch` は既定でスワップが 1GB 増えた時点で実行を落とす
(`--abort-delta-mb 0` で無効化、`--warn-delta-mb` で警告閾値)。

測定前には常駐アプリを落とし、できれば再起動する。ベースラインの空きメモリは
実験条件の一部なので `docs/16gb/runs/` に記録すること。
