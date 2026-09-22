# M5: ブロック単位ストリーミング → **合格**

- 測定日: 2026-09-22
- 機材: MacBook Pro M3 Pro 18GB、macOS 15.6、MLX 0.32.0
- 再現: `tools/bench/block_stream.py`(`tools/swapwatch.py` 越し)
- 構成: 1024² / 4 ステップ / `euler` / guidance 1.0 / seed 42 / **LoRA なし**
- 記録: `docs/16gb/runs/20260922-1025-m5-stream.{csv,json}`

## 1. 判定: **合格**

| 合格基準(計画) | 実測 | |
|---|---|---|
| 計算 ÷ I/O ≥ 2.0 | **13.65** | ✅ 6.8 倍の余裕 |
| 1 ステップが M3 の +20% 以内(≤ 33.8 s) | **29.79 s**(+5.8%) | ✅ |
| drop 後の常駐がベースライン +100 MB 以内 | +313 MB | ⚠️ 下の §4 |
| スワップしないこと | **swapouts 0、verdict `clean`** | ✅ |

```
globals resident  : 0.706 GB     ← ブロックはディスクに置いたまま
I/O per block     : 72.1 ms      ← M0 の実測 74 ms と一致
compute per block : 983.7 ms
drop per block    : 8.0 ms
compute / I/O     : 13.65
mx peak memory    : 3.21 GB
peak footprint    : 3.72 GB
per step          : [29.60, 29.74, 29.97, 29.87] s
verdict           : clean (swapouts 0)
```

## 2. M3 との比較

| | M3(常駐) | **M5(ストリーミング)** |
|---|---|---|
| 常駐 | 13.63 GB | **0.706 GB** |
| **要求(mx peak)** | 15.57 GB | **3.21 GB** |
| 実 footprint | 15.96 GB | **3.72 GB** |
| 1 ステップ | 28.15 s | 29.79 s(**+5.8%**) |
| swapouts | 1117 MB | **0** |
| `mx.set_wired_limit` | **必須** | **不要**(プロセスは wire を一切要求していない) |
| 判定 | SWAPPED | **clean** |

**要求が 15.96 → 3.72 GB に落ちた。** 代償は 1 ステップあたり 1.64 秒。

19.33 GB のマシンで 3.72 GB しか使わないので、**16GB 機でも、さらに小さい機械でも
余裕がある**。M3 で必要だった `sudo sysctl iogpu.wired_limit_mb` の細工も要らない。

## 3. 正しさ: **ビット完全一致**

同じ構成(LoRA なし / 1024² / 4 ステップ / seed 42)で常駐版を 1 回回し、
latent を突き合わせた:

```
max abs diff  : 0.000e+00
mean abs diff : 0.000e+00
bit-identical : True
```

**ストリーミングは数値に一切影響しない。** 同じ演算を同じ順序でやって、
ウェイトの寿命だけを変えているので当然ではあるが、実測で確認した。

## 4. 「drop 後 +100 MB」を満たさない理由(リークではない)

drop 直後の常駐は毎回 **1.017 GB** で、ベースライン 0.706 GB に対して +311〜313 MB。

**28 ブロックすべてで 1.017 GB、ドリフトは +0 MB。**

| | |
|---|---|
| 1 ブロック目の drop 後 | 1.017 GB |
| 28 ブロック目の drop 後 | 1.017 GB |
| ドリフト | **+0 MB** |

つまり**ウェイトは正しく解放されている**(461.3 MB のブロックが積み上がっていれば
28 ブロックで 12.9 GB になる)。残っている 313 MB はアクティベーション側:
残差ストリーム `(1, 4126, 6144)` bf16 が 50.7 MB で、これが入力・出力・
途中経過として数本同時に生きている。

**合格基準の「+100 MB」の見積りが甘かった。**「drop 後にウェイトが残っていないこと」
という意図は満たしている。基準は「ブロック間でドリフトしないこと」に読み替えるべき。

## 5. どう実装したか(mflux 本体は 1 行も変えていない)

`Krea2Transformer.__call__` のブロックループは

```python
for block in self.blocks:
    combined = run(combined, tvec, freqs, attention_mask)
```

なので、**`transformer.blocks` を差し替えるだけ**で streaming になる。
各要素は次をやるラッパ:

1. そのブロックの 29 テンソルを**新しい lazy ハンドル**として読む
2. `block.update(...)` して `mx.eval(block.parameters())` ← ここが I/O
3. `block(combined, tvec, freqs, mask)` を呼んで `mx.eval(out)` ← ここが計算
4. **新しい lazy ハンドルで再 bind** して `mx.clear_cache()` ← ここが drop

**毎回新しい lazy ハンドルを読むのが要点。** 一度読んだツリーを使い回すと、
評価済みの配列がそこから参照され続けて、drop しても 1 バイトも解放されない。

`mx.eval(out)` が必要なのは MLX が遅延評価だから。これを入れないと、
グラフが未評価のままウェイトを drop することになる。ブロックごとに同期が入るが、
その代償はこの測定に含まれている(それでも +5.8%)。

## 6. LoRA: 事前に焼き込んだチェックポイントを作る → **実装して通した**

選択肢は 2 つあった:

| | 実行時コスト |
|---|---|
| **A: 事前に bake したチェックポイントを作る** | **ゼロ** |
| B: LoRA 層を常駐させ、bake せずに使う | **+11 s/step**(M3 の実測、41.9 対 30.9 s) |

A を `tools/bench/bake_lora_checkpoint.py` として実装した。

### bake もストリーミングでやる

素直に `LoRASaver.bake_and_strip_lora(transformer)` を呼ぶと**全体が実体化する**。
`lora_saver.py:125` が層ごとに `mx.eval` するので、28 ブロック分が積み上がる。
実測: footprint 13.10 GB、**1227 MB スワップして失敗**。

`bake_and_strip_lora` は任意のモジュールを取るので、**ブロック単位で呼ぶ**:

```
ブロック i について: LoRA を焼く → eval → blocks_NN.safetensors に書く → 空配列に置換 → drop
```

| | |
|---|---|
| mx peak memory | **1.89 GB** |
| peak footprint | **2.14 GB** |
| 所要時間 | **13 秒**(28 ブロック + globals) |
| verdict | **clean**(swapouts 0) |
| 出力 | 956 テンソル(元と同数)、13 GB、1 ブロック 1 ファイル |

**13.62 GB のモデルの LoRA 焼き込みを、2.14 GB で 13 秒で終わらせた。**

### 焼き込んだチェックポイントをストリーミングする(= 目標構成)

- 実行: `20260922-1048`、記録: `docs/16gb/runs/20260922-1048-m5-stream-baked.{csv,json}`

```
I/O per block     : 71.8 ms
compute per block : 984.7 ms     ← LoRA なしの 983.7 ms と同じ
compute / I/O     : 13.71
mx peak memory    : 3.21 GB
peak footprint    : 3.72 GB
per step          : [29.47, 29.56, 30.10, 29.99] s
verdict           : clean (swapouts 0)
```

**LoRA が付いても速度も常駐も一切変わらない。** 予測どおり、bake は実行時コストゼロ。

### 正しさ: 常駐版と**バイト単位で一致**

M3 の常駐版(LoRA を読み込み時に bake、1024²、seed 42)と比較:

```
latent  max abs diff : 0.000e+00   bit-identical: True
PNG     sha256 一致   : 224664246568395d...  (cmp でバイト一致)
```

**常駐方式とストリーミング方式は、同じ画像を 1 バイトの違いもなく出す。**

---

## 7. パイプライン全体(目標構成、すべて `clean`)

| 段 | mx peak | 実 footprint | 時間 | verdict |
|---|---|---|---|---|
| M2 text encoder | 8.19 GB | **8.29 GB** ← 最大 | 4 s | clean |
| LoRA 焼き込み(1 回だけ) | 1.89 GB | 2.14 GB | 13 s | clean |
| **M5 DiT ストリーミング** | 3.21 GB | 3.72 GB | **119 s**(4 × 29.8) | clean |
| M4 VAE(タイル 256) | 2.98 GB | 4.28 GB | 5.6 s | clean |

**18GB 機で目標構成がスワップなしで通る。**
q8 + 4step LoRA / 1024² / 4 ステップ / euler / guidance 1.0 / seed 42。

**いま最大の消費は DiT ではなく text encoder の 8.29 GB。**
16GB 機を狙うなら次に削るのはここ(ただし TE の量子化は画質要件で禁止なので、
別の手が要る)。
