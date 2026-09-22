# M4: VAE だけのプロセスでデコードする → **実画像 1 枚**

- 測定日: 2026-09-22
- 機材: MacBook Pro M3 Pro 18GB、macOS 15.6、MLX 0.32.0
- 再現: `tools/bench/vae_decode.py`(`tools/swapwatch.py` 越し)
- 入力: [M3](2026-09-22-m3-dit-resident.md) が書いた 1024² の latent `[1, 16, 128, 128]`

## 1. 判定

**合格(タイルあり)。`clean`。**

| | タイルなし | **タイル 512** |
|---|---|---|
| `mx.get_peak_memory()` | 8.73 GB | **4.40 GB** |
| デコード時間 | 4.09 s | 4.70 s |
| swapouts | 170 MB | **0** |
| verdict | SWAPPED | **clean** |

1024² を一枚で復号すると、デコーダが 128×128×16 を 1024×1024×3 まで広げる過程で
**8.73 GB** 使う。VAE のウェイトは 0.51 GB しかないのに、である。
タイル 512 にすると 4.40 GB に落ちて、代償は 0.6 秒。

## 2. **これでパイプラインが端から端まで通った**

```
M2: text encoder 8.05 GB  → 埋め込み (1, 30, 30720)
M3: q8 DiT 13.62 GB       → latent [1, 16, 128, 128]
M4: VAE 0.51 GB           → 1024×1024 PNG
```

3 つを**別プロセスに分ければ**、どの段も単独では 18GB 機に載る。
出力: `docs/16gb/runs/images/20260922-0938-krea2-q8-4step-1024-tiled.png`

画像は意図どおり出ている。真鍮の腐食、窓の結露、被写界深度、作業台の木目まで
描けていて、**q8 + 4step 蒸留 LoRA が正しく動いていることが確認できた**。
これが q8 を譲れない要件にしている「描き込み」そのもの。

プロンプト(この計画で固定):

```
a photograph of a weathered brass diving helmet on a workshop bench,
morning light through a dusty window, shallow depth of field
```

seed 42 / 4 ステップ / `euler` / guidance 1.0 / 1024²。

## 3. 注意: VAE では `mx.get_peak_memory()` が実消費を大きく下回る

精密な `phys_footprint` で測ると:

| | `mx.get_peak_memory()` | `phys_footprint` |
|---|---|---|
| DiT (1024²) | 15.57 GB | 15.96 GB(差 0.39 GB) |
| **VAE (タイル 512)** | **4.40 GB** | **14.16 GB(差 9.8 GB)** |

DiT では一致するのに、VAE では 10 GB 近くずれる。畳み込み経路が MLX の
アロケータの外側(MPS 側)で確保しているためと思われる。

**この 14.16 GB は「VAE 単独のプロセスだから」通っている。**
M6 で 3 コンポーネントを 1 プロセスに統合するときは、
**DiT を破棄してから VAE を構築する**必要がある。`mx.get_peak_memory()` の
4.40 GB を見て「VAE は小さいから DiT と同居できる」と判断してはいけない。
