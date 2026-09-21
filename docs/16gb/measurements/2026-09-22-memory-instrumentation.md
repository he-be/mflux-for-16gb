# 計測の土台: macOS で MLX のメモリを何で測るか

- 測定日: 2026-09-22
- 機材: MacBook Pro M3 Pro 18GB、macOS 15.6 (Darwin 24.6.0)、MLX 0.32.0
- 再現: `tools/bench/memstat.py`、`tools/swapwatch.py`

この計画は「14.06 GB が載るか」を判定する。判定に使う数字が間違っていたら計画全体が
無意味になるので、先に計器を確かめた。**結論から言うと `ps rss` は使えない。**

## 1. `ps rss` は MLX の確保を 1 バイトも見ていない

4.00 GB の `mx.array` を確保して `mx.eval` し、同じプロセス内外から測った:

| 測り方 | 値 |
|---|---|
| `mx.get_active_memory()` | **4.00 GB** |
| `ps -o rss=` | **0.03 GB** |
| `resource.getrusage(RUSAGE_SELF).ru_maxrss` | **0.03 GB** |
| `footprint -p <pid>` → `phys_footprint` | **3.84 GB** |
| `vmmap --summary` → `IOAccelerator (graphics)` | 3.7 GB |

バッファの全域を GPU 側で読み切って(`mx.sum`)から測り直しても `ps rss` は 0.03 GB の
まま。MLX の確保は `IOAccelerator` リージョンなので、RSS の勘定に入らない。

つまり **`ps rss` は 14 GB 常駐しているプロセスを 30 MB と報告する。**

## 2. 使える計器

| 計器 | 何が分かるか |
|---|---|
| `mx.get_active_memory()` / `get_peak_memory()` / `get_cache_memory()` | MLX が握っている量。プロセス内からのみ |
| `footprint -p <pid>` の `phys_footprint` | プロセスの実フットプリント。**MLX の確保を含む** |
| 同 `phys_footprint_peak` | その **ピーク**。OS が覚えているので、サンプリング間隔が粗くても取り逃さない |
| `vm_stat` + `sysctl vm.swapusage` | システム全体。空き・compressor・swapouts |

`footprint -p` は 1 回 約 108 ms、root 不要(自分のプロセス)。ピークを OS 側が
保持しているので、`swapwatch` の 1 秒間隔でも高々一瞬の山を見逃さない。

## 3. 直した箇所

- `tools/swapwatch.py`: `child_rss_mb` を捨て、プロセスツリー全体の `phys_footprint` と
  `phys_footprint_peak` に置き換えた。**ツリー全体**なのは、実行が常に `uv run` 越しで、
  本体の python は孫プロセスになるため。直前の版は `uv run` の launcher(約 30 MB)を
  測っていて、8 GB 常駐の実行を「peak child RSS 0.03 GB」と報告した。
- `tools/bench/memstat.py` を新設。`hw.memsize` / `iogpu.wired_limit_mb` /
  GPU の working set / `vm_stat` / swap / メモリ圧 / 上位プロセスを 1 スナップショットにする。

## 4. 計画への影響

計画と HANDOFF の M3 / M5 の合格基準は「peak RSS」と書いてあったが、**その数字は
取れない**。判定は次の 2 本で行う:

1. `mx.get_peak_memory()`(プロセス内、MLX が握った量)
2. `swapwatch` の `phys_footprint_peak` と swapouts / compressor(システム側)

2 本が食い違ったら、どちらも信じないで原因を先に潰す。M2 では
8.19 GB(MLX ピーク)と 8.11 GB(footprint ピーク)で一致した。

## 5. 空きメモリの読み方

`memstat.py` の見出しの数字は `free` ではなく
**`free + inactive + speculative + purgeable`**(= claimable)。macOS は RAM の大半を
inactive / speculative に置いたまま要求に応じて返すので、`free` 単体は常に小さく出る
(実測: `free` 0.31 GB のとき claimable 7.53 GB)。

なお claimable が足りていても **compressor が膨らむ**ことがある。M2 の実行では
compressor が 1.26 → 5.69 GB に伸びた(swapouts は 0)。swapouts が出ていなければ
`clean` だが、これは「他のプロセスを圧縮して場所を作った」状態であって余裕ではない。
14 GB を測るときは再起動してから測る理由がここにある。
