# M0: Krea 2 Turbo q8 チェックポイントの物理配置

- 測定日: 2026-09-22
- 対象: `mflux-community/krea-2-turbo-mflux-q8`(`quantization_level: 8`, mflux 0.18.0 で保存)
- 出典: `model.safetensors.index.json`(推測なし、index そのもの)

計画 [M0](../../.cursor/plans/2026-09-22-krea2-block-streaming.md) の合格判定に使う事実。

## ブロックのシャード配置

- ブロック数 **28**、1 ブロックあたり **29 テンソル**(q8 なので `weight`/`scales`/`biases` を含む)。
- 配置は**ブロック番号の順**。1 ブロックは 1 シャードに収まるか、隣接 2 シャードにまたがるだけ。

| シャード | 含むブロック |
|---|---|
| `0.safetensors` | 0〜4(+ `first`) |
| `1.safetensors` | 4〜9 |
| `2.safetensors` | 9〜13 |
| `3.safetensors` | 13〜18 |
| `4.safetensors` | 18〜22 |
| `5.safetensors` | 22〜27 |
| `6.safetensors` | 27(+ `tmlp`, `tproj`, `txtmlp`, `txtfusion`, `last`) |

またがるのは block 4, 9, 13, 18, 22, 27 の 6 個だけで、いずれも隣接シャード。

## 判定

**合格**。デノイズはブロック 0→27 の順に進むので、読み出しも
ファイル先頭から末尾へ向かう連続アクセスになる。シークが支配する配置ではないため、
ブロック順に並べ直した専用ファイルを書き出す工程(M0b)は不要。

ブロック外の小物(`first`, `tmlp`, `tproj`, `txtmlp`, `txtfusion`, `last`)は
シャード 0 と 6 の両端にあり、常駐させる方針と整合する。

## 未測定(ダウンロード完了後に追記)

- ブロックごとの実バイト数(シャードのヘッダから)。
- 実シャードを `F_NOCACHE` で読んだときの帯域(合成ファイルでの 4.6〜5.2 GB/s と比較)。
