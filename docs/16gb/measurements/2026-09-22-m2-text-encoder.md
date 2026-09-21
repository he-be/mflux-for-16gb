# M2: text encoder だけのプロセスで埋め込みを作る

- 測定日: 2026-09-22
- 機材: MacBook Pro M3 Pro 18GB、macOS 15.6、MLX 0.32.0
- 再現: `tools/bench/te_encode.py`(`tools/swapwatch.py` 越し)
- 対象: `mflux-community/krea-2-turbo-mflux-q8` の `text_encoder/`(Qwen3-VL-4B, bf16)
- 測定条件: **クリーンな状態ではない**(常駐アプリあり、claimable 7.5〜11.0 GB、
  swap に既存の 492 MB)。M1 の再起動後測定はまだ行っていない。

## 1. 判定

**合格。`clean`。**

同じ実行を 3 回。数字は 3 回とも一致した。

| | 値 |
|---|---|
| `mx.get_active_memory()` | **8.05 GB**(= チェックポイントの TE 実サイズ 8.045 GB) |
| `mx.get_peak_memory()` | 8.19 GB |
| `mx.get_cache_memory()` | 0.15 GB |
| `phys_footprint_peak`(ツリー全体) | 8.11 / 8.29 GB |
| swapouts | **0 ページ** |
| swap 増分 | **0 MB** |
| verdict | **clean** |

MLX 側(8.19 GB)と OS 側(8.11〜8.29 GB)が一致したので、どちらの計器も信じられる。
計器そのものの検証は [計測の土台](2026-09-22-memory-instrumentation.md) に分けた。

## 2. 時間

| 段 | 時間 |
|---|---|
| モジュール構築 | 0.003 s |
| `mx.load`(lazy、mmap のみ) | 0.003 s |
| **実体化**(`mx.eval`、8.05 GB を読む) | **1.30〜1.41 s** = 約 5.9 GB/s |
| tokenizer ロード | 1.45 s(初回のみ 4.26 s) |
| **エンコード**(36 層 × 30 トークン) | **0.63〜0.81 s** |
| 保存 | 0.002 s |

実体化の 5.9 GB/s は M0 の実測帯域(全体読みで 4.47 GB/s、拾い読みで 6.24 GB/s)と
同じ桁で、別経路の sanity check として合う。

## 3. 出力

| | |
|---|---|
| 形 | `(1, 30, 30720)` bfloat16 = **1.84 MB** |
| 30720 | 12 tap 層 × hidden 2560(`KREA2_TAP_LAYERS`) |
| 30 | チャットテンプレートの system + user 開始を落とした後のトークン数 |
| 置き場所 | `~/Library/Caches/mflux/16gb-bench/krea2-embeds.safetensors` |

メタデータにプロンプト・モデルパス・tap 層を埋めてあるので、M3 側で取り違えない。
プロンプト(以降この計画で固定):

```
a photograph of a weathered brass diving helmet on a workshop bench,
morning light through a dusty window, shallow depth of field
```

guidance は 1.0 なので negative の埋め込みは作っていない(単一パス)。

## 4. ここで分かった注意点

- **`mx.load` は lazy。** `mx.eval(model)` を明示しないと、読み出しコストは次に触った
  処理の時間に紛れる。`te_encode.py` は実体化を別枠で測るために明示的に eval している。
- **compressor が 1.26 → 6.21 GB に伸びた。** swapouts は 0 なので判定は `clean` だが、
  8 GB の場所は「他のプロセスを圧縮して」作られている。この状態のまま 14 GB を測ると
  条件が汚いので、M3 は再起動後に測る。
- **TE は量子化されていない。** 定義が `skip_quantization=True`(量子化すると条件付けが
  劣化する)なので、q8 チェックポイントでも TE だけは bf16 8.05 GB のまま。ここは
  削れない。

## 5. これで確定したこと

TE (8.05 GB) と DiT (13.62 GB) を別プロセスにすれば、**TE 側は 18GB 機で余裕で通る**。
残る問題は DiT + LoRA = 14.06 GB 側だけになった(M3)。
