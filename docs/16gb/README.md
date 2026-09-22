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
| `tools/bench/` | 段階ごとの計測スクリプト(`memstat.py` / `te_encode.py` / `dit_steps.py` / `vae_decode.py` / `block_stream.py` / `bake_lora_checkpoint.py`) |
| `~/Library/Caches/mflux/16gb-bench/` | 段の間で受け渡す中間生成物(埋め込み・latent)。リポジトリには入れない |

## 索引

- **[引き継ぎ(2026-09-22)](HANDOFF.md)** ← 新しいセッションはここから

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
- [計画: ブロック単位ウェイトストリーミング](../../.cursor/plans/2026-09-22-krea2-block-streaming.md)

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
| text encoder | 8.19GB | **8.29GB** ← 最大 | 4 s |
| LoRA 焼き込み(1 回だけ) | 1.89GB | 2.14GB | 13 s |
| **DiT ストリーミング** | 3.21GB | 3.72GB | **119 s** |
| VAE(タイル 256) | 2.98GB | 4.28GB | 5.6 s |

q8 + 4step LoRA / 1024² / 4 ステップ / euler / guidance 1.0 / seed 42。
画像: `docs/16gb/runs/images/`。

**いま最大の消費は DiT ではなく text encoder の 8.29GB。**
16GB 機を狙うなら次に削るのはここ。

**まだやっていないこと**

- **M6**: この方式を mflux 本体へ入れる(いまは `tools/bench/` の分割プロセス版のみ)。
- **M7**: 2 回実行して画像一致の確認、1280²(LoRA の学習 σ と一致する解像度)。

## 実行のしかた

現状の mflux は 3 コンポーネントを同時に載せるので、この構成 (22.2GB) は 18GB 機では
そのまま走らない。**いま動くのは `tools/bench/` の 3 プロセス分割版**
(引き継ぎの §5 にコマンドがある)。下は段階ロード / ストリーミングが入った後の形。

```sh
uv run python tools/swapwatch.py --csv docs/16gb/runs/$(date +%Y%m%d-%H%M)-q8-4step.csv -- \
  uv run mflux-generate-krea2 \
    --model ~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/<rev> \
    --prompt "..." --seed 42 --steps 4 --scheduler euler \
    --width 1024 --height 1024 \
    --lora-paths 'lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors' \
    --lora-scales 1.0 --low-ram
```

`swapwatch` は既定でスワップが 1GB 増えた時点で実行を落とす
(`--abort-delta-mb 0` で無効化、`--warn-delta-mb` で警告閾値)。

測定前には常駐アプリを落とし、できれば再起動する。ベースラインの空きメモリは
実験条件の一部なので `docs/16gb/runs/` に記録すること。
