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
| `docs/16gb/runs/` | 実行ログ(swapwatch の CSV、コマンド、所要時間) |
| `.cursor/plans/` | 実装計画(RULE.md の規約) |
| `tools/` | fork 固有のツール |

## 索引

- [Krea 2 を 16GB / 18GB で動かすための調査](research/krea2-low-memory-mac.md)
- [Krea 2 Turbo 4-step 蒸留 LoRA の使い方](research/krea2-4step-lora.md)
- [M3 Pro 18GB: 計算律速と SSD ストリーミングの実測](measurements/2026-09-22-m3pro-compute-vs-ssd.md)（合成ベンチ。実ウェイトでは未測定）
- [M0: Krea 2 Turbo q8 チェックポイントの物理配置](measurements/2026-09-22-krea2-q8-checkpoint-layout.md)
- [計画: ブロック単位ウェイトストリーミング](../../.cursor/plans/2026-09-22-krea2-block-streaming.md)

## いま分かっていること(2026-09-22)

**実測済み(事実)**

- このマシン: `hw.memsize` 19.33GB、`iogpu.wired_limit_mb` 14.34GB、
  `max_recommended_working_set_size` 15.03GB。
- q8 の実サイズ: DiT 13.62GB(1 ブロック 461.3MB × 28 + globals 0.71GB)、
  TE 8.05GB、VAE 0.51GB。**DiT + 4step LoRA = 14.06GB** が載るかどうかが勝負どころ。
- 実ウェイトの読み出し: 1 ブロック 461.3MB を **74 ms (6.24 GB/s)**、
  transformer 全体 13.62GB を 3.05 s (4.47 GB/s)。散らばった配置でも帯域は落ちない。
- M3 Pro の行列積は 5.1〜5.9 TFLOPS(合成テンソル)。
- 合成 safetensors 28 個 × 138MB を毎周ディスクから流し直しても、常駐 0.17GB で
  計算律速のまま(bind→eval→drop でメモリは戻る)。

**まだ仮説(Krea 2 の実ウェイトでは未測定)**

- **q8 DiT を 18GB 機に常駐させられるか。** クリーンな状態(再起動直後、常駐アプリ無し)で
  14.06GB + アクティベーションが載るかは未測定。プロセスを分けて測る(計画 M2〜M4)。
- **ブロック単位ストリーミングで I/O が隠れるか。** 読み出し側は 74 ms/ブロックと
  確定したが、計算側(実ブロックの forward 時間)が未測定。比が 2.0 を超えるかが
  分かれ目(計画 M5)。1 ステップ 約 29 秒という数字は FLOPS からの割り算であって
  実測ではない。

## 実行のしかた

現状の mflux は 3 コンポーネントを同時に載せるので、この構成 (22.2GB) は 18GB 機では
そのまま走らない。下は段階ロード / ストリーミングが入った後の形。

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
