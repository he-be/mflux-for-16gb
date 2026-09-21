# 計画: Krea 2 のブロック単位ウェイトストリーミング

- 日付: 2026-09-22
- ブランチ: `feat/krea2-block-streaming`
- 管理先: fork [`he-be/mflux-for-16gb`](https://github.com/he-be/mflux-for-16gb)(upstream へ PR は出さない)
- 根拠データ: [`docs/16gb/measurements/2026-09-22-m3pro-compute-vs-ssd.md`](../../docs/16gb/measurements/2026-09-22-m3pro-compute-vs-ssd.md)

## 目標

M3 Pro 18GB で **q8 + 4step LoRA** の生成を、スワップ 0 で通す。
常駐を「全ブロック」から「1〜2 ブロック + アクティベーション」に落とす。

検証可能なゴール:

1. `tools/swapwatch.py` の判定が `clean`(swapouts 増加 0)。
2. ストリーミング on/off で、同一シード・同一プロンプトの出力画像が一致する
   (数学は変わらないので、ずれたら実装バグ)。比較は q8 が載る機材、または
   ストリーミング側だけを 2 回実行しての自己再現性で代替。
3. 1 ステップあたりの時間が、非ストリーミング比で +20% 以内(I/O が隠れていることの確認)。

## 現状

`Krea2Initializer.init` は 3 コンポーネントを構築して `mx.eval(model)` で一括実体化する
(`krea2_initializer.py:30-38`)。破棄側は `MemorySaver` が持っていて、エンコード直後に TE を、
`--low-ram` ならループ後にトランスフォーマーを捨てる(`memory_saver.py:77`)。
足りないのは「まだ使わないブロックを実体化しない」こと。

MLX の `mx.load` は lazy handle を返し、参照を捨てれば解放される(計測 §3)。
つまり必要なのは新しいローダーではなく、**bind / eval / drop の順序制御**。

## 設計

### `Krea2WeightStream`(新規, `src/mflux/models/krea2/weights/krea2_weight_stream.py`)

- 構築時に safetensors のパス群だけを保持する(mflux セーブ形式のシャード + index)。
- `bind(block_index)`: そのブロックのテンソルを lazy handle として `tree_unflatten` し、
  `block.update(...)` する。量子化済みチェックポイントなら `weight`/`scales`/`biases` を
  そのまま流す(再量子化しない)。
- `release(block_index)`: 実体化済みの配列への参照を捨て、lazy handle に戻す。
- `prefetch(block_index)`: `mx.async_eval` で次ブロックの実体化を先行させる(第 2 段階)。
- ブロック外の小物(`first`, `tmlp`, `tproj`, `txtfusion`, `last`)は常駐のまま。

### フック位置

- `Krea2Transformer.__call__` の 28 ブロックのループに、bind → 実行 → release を差し込む。
  ストリームが `None` のときは現状の挙動のまま(既定は非ストリーミング)。
- `Krea2Initializer.init`: ストリーミング有効時はトランスフォーマーの重みを
  `mx.eval` せず、構造(QuantizedLinear の形)だけ作ってストリームを持たせる。
- LoRA: `--no-bake-lora` の runtime adapter は rank 64 = 0.44GB なので常駐のまま扱う。
  bake は bind 後のブロックに対して行う必要があり、ステップごとに再計算になるので
  ストリーミング時は bake を禁止(明示エラー)。

### CLI

- `--stream-weights`(既定オフ)。`--low-ram` からも有効化するかは実測後に決める。
- 進捗表示は既存の `MemorySaver` のメモリ統計に相乗り。

## 変更ファイル

| ファイル | 変更 |
|---|---|
| `src/mflux/models/krea2/weights/krea2_weight_stream.py` | 新規 |
| `src/mflux/models/krea2/model/krea2_transformer/transformer.py` | ブロックループにフック |
| `src/mflux/models/krea2/krea2_initializer.py` | ストリーミング時は eval を遅延 |
| `src/mflux/models/krea2/variants/txt2img/krea2.py` | ストリーム生成とライフサイクル |
| `src/mflux/cli/parser/parsers.py` | `--stream-weights` |
| `tools/swapwatch.py` | 済(スワップ見張り) |
| `docs/16gb/` | 計測・運用の記録 |

## 段階

1. **PoC**(mflux 本体は触らない): スクリプトで Krea2Transformer をストリーミング実行し、
   1 ステップ時間と常駐を実測する。ここで bind/drop のオーバーヘッドを確認。
2. **本実装**: 上記の変更。まず q8 + 4step LoRA、1024²、4 ステップ、seed 42 で通す。
3. **TE**: エンコードは 1 回きりで計算量が小さいので、ストリーミングより
   「load → encode → 破棄」で足りるか実測してから決める。
4. **記録**: `docs/16gb/runs/` に swapwatch の CSV とコマンドを残す。

## 非目標

- upstream への PR、他モデルへの一般化(まず Krea 2 だけ)。
- 画質の評価(ストリーミングは数学を変えないので、画質は別の軸)。

## リスク

- bind を 28 回/ステップ行うオーバーヘッドが無視できない場合 → 2 ブロック分の
  ダブルバッファに切り替える。
- MLX のキャッシュアロケータが解放を遅延させる場合 → ブロックごとに `mx.clear_cache()`、
  あるいは `--mlx-cache-limit-gb` を小さくする。
- 量子化済みチェックポイントの `scales`/`biases` を含む部分更新で
  `nn.QuantizedLinear` の内部整合が崩れる場合 → 形だけ先に作ってから
  `update` する順序を守る(`WeightApplier` の stored-quantization 経路と同じ作り)。
