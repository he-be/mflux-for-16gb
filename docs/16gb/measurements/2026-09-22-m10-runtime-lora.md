# M10: ストリーミング経路に実行時 LoRA(2026-09-22 夜)

機材: Mac mini (Apple M6) 16GB。1024² / 4 ステップ / euler / guidance 1.0 / seed 42 /
`--block-streaming`、モデルは `krea2-lowram`(4step LoRA 焼き込み済み + q8 TE)。
実行は `tools/swapwatch.py` 越し。**2 本とも同じセッションで連続して取った。**

## 結論: 合格。焼き込み不要で LoRA を差し替えられる

| | LoRA なし | **LoRA あり**(rank 64 / 456 キー / 228 層) |
|---|---|---|
| 1 ステップ | 7.92 s | **10.32 s** (+30%) |
| 計算 / ブロック | 282.2 ms | **363.9 ms** (+81.7 ms) |
| bind / ブロック | 0.2 ms | 1.8 ms |
| プリフェッチ / ブロック | 139.5 ms | 141.0 ms(変わらない) |
| 計算 ÷ 読み | 2.02 | 2.58 |
| mx peak | 4.40 GB | 4.59 GB |
| **実 footprint** | **5.16 GB** | **6.62 GB** (+1.46 GB) |
| swapouts | 0 | **0** |
| verdict | `clean` | **`clean`** |

ログ: [`runs/m6mini/20260922-1930-m10-base.csv`](../runs/m6mini/20260922-1930-m10-base.csv) /
[`…-m10-lora.csv`](../runs/m6mini/20260922-1930-m10-lora.csv)。

**当てたのは 4step LoRA そのもの**(`krea2_turbo_4step_rank_64_lora_comfyui.safetensors`,
scale 0.5)。焼き込み済みのスナップショットにもう一度重ねているので、
**画像 `images/20260922-1930-m10-lora-doubled.png` は二重適用で参照にならない**。
採ったのは時間とメモリだけ。一方でこれは **456 キー全層に当たる最も重いケース**で、
attention だけの style LoRA はこれより安い。

## 1. なぜ以前は併用できなかったか

`Krea2Initializer._init_staged` が `--lora-paths` を明示的に拒否していた。理由は正しく、
**ストリーミングされたブロックには焼き込む先が無い**。ブロックの重みは毎ステップ
ディスクから来て捨てられるので、`LoRASaver.bake_and_strip_lora` が畳み込んだ結果は
次の bind で上書きされる。

## 2. 直した形: 焼き込まず、側道として当てる

`bake_lora=False` で当てると `LoRALinear` が量子化層を包み、`base(x) + scale·(x·A·B)` になる。
A/B は**チェックポイントに無い**ので bind で上書きされず、ブロックに残り続ける。
つまり**ストリーミングと矛盾しない唯一の当て方が、たまたま一番安い当て方でもある**。

踏んだのは 2 か所。

### 2.1 アダプタは層を 1 段深くする

包んだ瞬間、チェックポイントの `blocks.7.attn.wq.weight` が生きたブロックでは
`attn.wq.linear.weight` に移る(`FusedLoRALinear` なら `base_linear`)。
`Krea2BlockStream` は読み出し位置を付け替える(`_position` / `_wrapped_paths`)。
これは `mx.load` 経路と**先行確保したバッファへの直接読み**の両方に効く必要がある
——後者はテンソル名でバッファを引くので、名前がずれると黙って別の場所へ書く。

### 2.2 `mlp.down` の K4 分割はアダプタの下で保たないといけない

`_down_in_slices` は `isinstance(self.down, nn.QuantizedLinear)` で守られていたので、
包んだ途端に素通りして M9c で消したはずの崖(92 → 51 ms)が戻る。
**分割は素の量子化層に当て、アダプタの寄与を側道として足し戻す**。
`Krea2BlockStream._down_adapter` が `(素の層, delta)` を組み立てて
`Krea2SwiGLU._down_adapter` に置く。`delta` が基底の出力を必要としない形で書ける
アダプタ(LoRA、LoRA だけの Fused)にだけ付け、**LoKr の dora は基底の重みごと
スケールするのでこの形が無く、分割なしの経路に落とす**。

## 3. 増分の内訳

+81.7 ms/ブロックは FLOPs では説明できない。rank 64 の側道は 1 ブロックあたり
16 本で、合計 51 GFLOP ——ブロックの 4 TFLOP に対して 1.3%。
実際に増えているのは**細長い matmul のカーネル起動と、bf16 の A/B と q8 の基底が
別カーネルに落ちる分**。M9b の予算表で言えば「matmul 単体和 256 ms」の外側。
縮めるなら A/B をブロック内で 1 本のバッチ matmul にまとめる手があるが、
**M8c の教訓どおり本番形(プリフェッチあり)で測ってから**にすること。

## 4. 宿題

- **footprint の絶対値が M9c の記録(6.18 GB)と食い違う。** 今日の LoRA なしは
  5.16 GB。設定は同じはず。`phys_footprint` は swapwatch が約 2 Hz で
  サンプリングしているので瞬間ピークを取りこぼしうるが、1 GB の差は大きい。
  **同一セッションで取った 2 本の差(+1.46 GB)は信用してよいが、絶対値は要再測定。**
- +1.46 GB も A/B の実サイズ(rank 64 で約 0.5〜0.7 GB)より大きい。
  mmap した LoRA ファイル(438 MB)のページと側道の中間バッファが乗っていると
  思われるが、内訳は詰めていない。
- 画質の比較をしていない。**style LoRA を 1 本入れて、焼き込み版と実行時版で
  同じ絵が出ることを確認する**のが次。
