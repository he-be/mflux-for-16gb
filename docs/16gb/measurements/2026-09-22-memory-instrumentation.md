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


---

# 追記(同日、M3 の途中で判明): `footprint -p` は大きなプロセスを GB に丸める

## 何が起きたか

M3 の全段で、`swapwatch` が報告する peak footprint が条件によらず同じ値になった:

```
13326 MB / 14350 MB / 15374 MB
```

これを「OS がプロセスに許す上限に当たっている」と解釈し、
**「このマシンの 1 プロセスあたりの実効上限は 16.12 GB」という結論を文書に書いた。
それは誤りだった。**

3 つの値は **きっかり 1024 ずつ離れている**。分解すると:

```
13326 = 13 * 1024 + 14
14350 = 14 * 1024 + 14
15374 = 15 * 1024 + 14
```

`footprint -p` は、プロセスが大きくなると **`phys_footprint: 13 GB` のように
整数 GB で出力する**。そこに `uv run` launcher の 14 MB が足されていた。
つまり観測していたのは「13 GB 台」「14 GB 台」「15 GB 台」という丸めであって、
同一の上限ではなかった。**同じに見えたのは、丸めたから。**

4 GB 級のプロセスでは MB 単位で出る(`phys_footprint: 3914 MB`)ので、
最初の検証では気づけなかった。

## 直し方: `proc_pid_rusage` を直接読む

`libSystem` の `proc_pid_rusage(pid, RUSAGE_INFO_V0, &buf)` は
`ri_phys_footprint` をバイト単位で返す。構造体は 16 バイトの uuid に続いて
uint64 が並び、`ri_phys_footprint` は 8 番目なので **オフセット 72**。

```python
_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
buf = (ctypes.c_uint8 * 96)()
_LIBC.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(0), ctypes.byref(buf))
phys_footprint = int.from_bytes(bytes(buf[72:80]), "little")
```

4 GB を保持するプロセスで検証: **4103570944 B** に対しツールは **3914 MB**
(= 3914 MiB = 4103565312 B)。一致する。

サブプロセスを起こさないので 108 ms → ほぼ 0 になり、ピークは
`phys_footprint_peak` が取れなくなる代わりにサンプルの最大値で取る
(精密なピークはプロセス内の `mx.get_peak_memory()` が持っている)。

## 教訓

- **同じ数字が繰り返し出たら、まず計器を疑う。** 「3 回とも 1 MB 単位で同じ」は
  物理現象ではなく、丸めの症状だった。
- ツールの出力は**単位だけでなく桁数も**確認する。同じツールが値の大きさで
  MB と GB を切り替えることがある。
- 幸い、**この誤りは M3 の判定(スワップする / しない)を変えていない。**
  判定は swapouts と `mx.get_peak_memory()` で出しており、そちらは正しい。
  変わったのは「なぜ」の説明のうち 1 つだけ。

## 補足: VAE では MLX の会計が実消費を大きく下回る

精密化した計器で M4 を測ると、**`mx.get_peak_memory()` 4.40 GB に対して
`phys_footprint` が 14.16 GB** まで伸びる(タイル 512 でデコード)。
DiT 側では mx peak 15.57 GB に対し footprint 15.96 GB(差 0.25 GB = python + MLX)で
一致しているので、これは VAE 固有。畳み込み経路が MLX のアロケータの外側で
確保しているためと思われる。**畳み込みを含むモデルで `mx.get_peak_memory()` だけを
見て容量を判断しない。**
