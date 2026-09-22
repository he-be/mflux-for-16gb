# M1: クリーンな状態(再起動直後)のベースライン

- 測定日: 2026-09-22 08:59(再起動直後)
- 機材: MacBook Pro M3 Pro 18GB、macOS 15.6、MLX 0.32.0
- 再現: `tools/bench/memstat.py --label clean-boot`
- 記録: `docs/16gb/runs/20260922-0859-m1-baseline.json`

## 1. 一番大事な発見: `iogpu.wired_limit_mb` は 0 が既定だった

```
                                  再起動前(前セッション)   再起動後(既定)
iogpu.wired_limit_mb              14336 MB                 0 MB
max_recommended_working_set_size  15.03 GB                 12.88 GB
```

これまでの文書は `iogpu.wired_limit_mb = 14336` をこのマシンの既定として記録していたが、
**それは前のセッションが `sysctl` で手で上げた値**で、再起動で消える。既定は 0。

そして 0 のとき **MLX に推奨される working set は 12.88 GB** しかない。
これは **DiT + LoRA の 14.06 GB より小さい**。前提が変わる:

- 14336 に上げると working set は 15.03 GB に伸びる(前セッションの実測値がそれ)。
- つまり計画の段階 3(`sudo sysctl iogpu.wired_limit_mb` を上げる)は
  「最後の手段」ではなく、**14.06 GB を載せるための前提条件**の可能性が高い。

`max_buffer_length` は 9.66 GB で wired limit に関係なく一定。1 テンソルは最大
100MB 程度なので、ここは制約にならない。

## 2. ページの内訳(再起動直後)

| キュー | |
|---|---|
| free | 0.06 GB |
| active | 8.19 GB |
| inactive | 4.50 GB |
| speculative | 3.87 GB |
| wired | 1.85 GB |
| purgeable | 0.42 GB |
| compressor | **0.00 GB** |
| swap used | **0 MB** |

- compressor 0 / swap 0 なので、状態としては本当にクリーン。
- ただし **claimable(free + inactive + speculative + purgeable)は 8.85 GB しかない。**
  再起動直後は Spotlight の索引作成(`corespotlightd` 0.55 GB、`mds_stores`、
  `mediaanalysisd`)が走っていて、`active` 8.19 GB の多くは起動時に読んだファイルの
  キャッシュ。

## 3. claimable は上限ではない

M2(TE 8 GB)を 1 回通した直後に測り直すと:

| | 再起動直後 | M2 実行後 |
|---|---|---|
| free | 0.06 GB | **8.23 GB** |
| claimable | 8.85 GB | **11.96 GB** |
| compressor | 0.00 GB | 0.80 GB |

M2 の実行が `active` に居たクリーンなファイルキャッシュを追い出した結果、
**起動直後より広くなった**。

つまり `active` の中身もかなりの部分が回収可能で、**claimable は下限の目安**。
「claimable < 14.06 GB だから載らない」とは言えない。実測するしかない。

## 4. 以降の実験条件

M3 以降はこの状態を基準にする:

```
hw.memsize                        19.33 GB
iogpu.wired_limit_mb              0 (既定)
max_recommended_working_set_size  12.88 GB
swap used                         0 MB
compressor                        0 GB
claimable                         8.85 GB (起動直後) / 11.96 GB (キャッシュ追い出し後)
```
