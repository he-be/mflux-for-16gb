# mflux-for-16gb

16GB / 18GB の Apple Silicon で mflux の大きいモデル(まずは Krea 2)を
**スワップさせずに**動かすための fork。

- upstream: [`mflux-community/mflux`](https://github.com/mflux-community/mflux)
- この fork: [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)
- **upstream へ PR は出さない。** 作業はこの fork で完結させる。

## 方針

1. **計測してから決める。** 見積りだけで「載らない」と結論しない。数字は
   `docs/16gb/measurements/` に日付つきで残す。
2. **スワップは失敗とみなす。** 実行は必ず `tools/swapwatch.py` 越しに行い、
   判定が `clean` でない実行結果は採用しない。
3. **upstream のファイルはできるだけ触らない。** 取り込み時の衝突を減らすため、
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

- M3 Pro の行列積は 5.1〜5.9 TFLOPS、内蔵 SSD は F_NOCACHE で 4.6〜5.2 GB/s。
- 合成 safetensors 28 個 × 138MB を毎周ディスクから流し直しても、常駐 0.17GB で
  計算律速のまま(bind→eval→drop でメモリは戻る)。
- q8 チェックポイントは 28 ブロックがブロック順に並び、連続読みできる配置。
- 全ブロック常駐を前提にすると、18GB 機(working set 15.03GB)では q3 しか載らない。

**まだ仮説(Krea 2 の実ウェイトでは未測定)**

- 「Krea 2 の推論は計算律速なので、ブロック単位のストリーミングで I/O が隠れる」
  — 1 ステップ 約 29 秒という数字は FLOPS からの割り算であって実測ではない。
  実ブロック 1 個を測る [M1](../../.cursor/plans/2026-09-22-krea2-block-streaming.md)
  が go/no-go。計算 ÷ I/O が 2.0 を下回ればこの方針は捨てる。
- したがって「RAM 容量は制約ではない」も、まだ確定していない。

## 実行のしかた

```sh
uv run python tools/swapwatch.py --csv docs/16gb/runs/$(date +%Y%m%d-%H%M)-q8-4step.csv -- \
  uv run mflux-generate-krea2 \
    --model ~/.cache/huggingface/hub/models--mflux-community--krea-2-turbo-mflux-q8/snapshots/<rev> \
    --prompt "..." --seed 42 --steps 4 --scheduler euler \
    --width 1024 --height 1024 \
    --lora-paths 'lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors' \
    --lora-scales 1.0 --no-bake-lora --low-ram
```

`swapwatch` は既定でスワップが 1GB 増えた時点で実行を落とす
(`--abort-delta-mb 0` で無効化、`--warn-delta-mb` で警告閾値)。
