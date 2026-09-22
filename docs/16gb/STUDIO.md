# studio — ブラウザから使う

CLI を覚えずに Krea 2 を回すための薄い Web UI。**計算は M6 mac mini、操作は MacBook Pro**
という使い方を前提にしている。

## なぜ自作なのか

upstream の mflux は CLI と Python API だけで GUI を持たない。第三者製の GUI は
[README の Related projects](../../README.md#related-projects) に並んでいる
(Mflux-ComfyUI / MLXBits Image Studio / MFLUX-WEBUI ほか) が、**どれも upstream の
mflux を呼ぶので `--block-streaming` と低メモリスナップショットを知らない**。
それ無しで q8 Krea 2 を回すと要求 15.96 GB でスワップする
([M3](measurements/2026-09-22-m3-dit-resident.md))。16 GB の mini では動かない。

studio は `mflux-generate-krea2` を**生成のたびに 1 プロセス起動するだけ**にしてある。
この fork のメモリの根拠は全部そのコマンドで測ったものなので、
プロセスを使い回して MLX のバッファを跨がせると測定が無効になる。
代償は 1 枚あたり約 8 秒の起動コスト(1024² で 42 秒の実行に対して)。

依存は**標準ライブラリだけ**。`tools/studio/server.py` と `index.html` の 2 ファイル。

## 使いかた

### mini で動かして MBP から使う(通常)

```sh
just studio-mini
```

MBP のチェックアウトを mini に rsync し、mini で studio を起動し、MBP のブラウザを開く。
バインドは **Thunderbolt bridge の 169.254.24.129 だけ**(Wi-Fi 側には出ない)。

```sh
just studio-mini-log     # mini 側のログを追う
just studio-mini-stop    # 止める
```

### MBP だけで動かす

```sh
just studio
```

mini が繋がっていないときはこちら。1024² で 22.9 s/step なので 1 枚 100 秒くらいかかる。

### 前提

`~/Library/Caches/mflux/16gb-bench/krea2-lowram` が両機にあること
([README の 一回だけの前処理](README.md#一回だけの前処理))。無ければ studio は起動時に
その旨を出して終了する。

## 画面

| | |
|---|---|
| プロンプト | ⌘Enter で生成 |
| サイズ | 1024² / **1280²(LoRA の学習 σ と一致、描き込みが多い)** / 縦横 / 768² |
| ステップ・guidance・枚数 | 既定は 4 / 1.0 / 1。枚数はシードを 1 ずつ進めて連番で積む |
| シード | 空欄でランダム。ギャラリーから拾い直せる |
| LoRA | パスと強さ。行を足せば重ねられる |
| 実行中 | 段・ステップ・経過・s/step・残り時間、中止ボタン |
| ギャラリー | 新しい順。クリックで拡大、**設定を読み込む**・保存・削除 |

画像は生成した機械の `~/Pictures/mflux-studio/` に溜まる。設定は同じ名前の
`.studio.json` に書いてあるので、あとから同じ条件を再現できる。

**キューは 1 本。** GPU は 1 つしかないので、2 枚同時に走らせても両方が遅くなるだけ。

## LoRA

`--block-streaming` と LoRA は**以前は併用不可**だった(ブロックがディスクにあるので
焼き込めない)。いまは**焼き込まずに側道として当てる**ので、UI から自由に差し替えられる。
チェックポイントを焼き直す必要はない。

実装は [M10](measurements/2026-09-22-m10-runtime-lora.md)。要点だけ:

- 常駐が増えるのは LoRA の A/B とその中間バッファだけ(実測 +1.46 GB)。ブロック本体は
  今までどおり毎ステップ読み直す。
- アダプタは層を 1 段包むので、チェックポイントの `attn.wq.weight` は生きたブロックでは
  `attn.wq.linear.weight` に移る。`Krea2BlockStream` が読み出し位置を付け替える。
- `mlp.down` の K4 分割は**アダプタの下で維持される**。分割は素の量子化層に当て、
  アダプタの寄与を側道として足し戻す。

### 実測(M6 mini 16GB / 1024² / 4 ステップ)

| | LoRA なし | LoRA あり(rank 64、456 キー) |
|---|---|---|
| 1 ステップ | 7.92 s | **9.25 s** (+17%) |
| 計算 / ブロック | 282.2 ms | **328.1 ms** |
| 実 footprint | 5.16 GB | **5.79 GB** |
| swapouts | 0 | **0** (verdict `clean`) |

測ったのは 4step LoRA そのもの(**全 456 キー = 最も重いケース**)。attention だけの
軽い style LoRA ならこれより安い。

**この +17% を 0 にしたいときは焼き込む**([M10b](measurements/2026-09-22-m10b-lora-budget.md) §4)。
`bake_lora_checkpoint.py` は `krea2-lowram` の上にもう 1 枚重ねられて **15 秒 / 13 GB**、
焼いた後は LoRA なしと同じ 7.92 s/step。1 枚あたり 5.3 秒得なので **3 枚で元が取れる**。
scale を変えると焼き直しになるので、**試行錯誤は実行時、決まったら焼く**。

## 注意

- **`--lora-paths` の旧エラーは無くなった。** 焼き込み済みスナップショットに
  さらに LoRA を重ねる形になるので、4step LoRA をもう一度当てると二重になる。
  UI から足すのは **style LoRA だけ**にすること。
- studio は swapwatch を通していない。**数字を採る実行は今までどおり
  `tools/swapwatch.py` 越しに**やる(README の 生成 の節)。studio は日常用。
- バインド先を `0.0.0.0` にすると LAN に出る。`just studio-mini` は
  Thunderbolt bridge のアドレスだけに縛ってある。
