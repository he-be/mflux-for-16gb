# Krea 2 を 16GB / 18GB の Mac で動かすための調査メモ

- 調査日: 2026-09-21 〜 22
- 対象コミット: `ada5323`(mflux 0.20.0 直後の main)
- 環境: Apple M3 Pro 18GB、MLX 0.32.0
- 前提にするターゲット: M5 以降の 16GB 機

このメモは、Krea 2 のウェイトの入手元、mflux の量子化とメモリ節約の実装、
M5 の Neural Accelerator と量子化形式の関係、他プロジェクトの状況、
そして 16GB 機で動かすために必要な作業をまとめたものです。

**実測していない値について**: メモリ量はすべて Hugging Face API のファイルサイズからの
計算値です。Krea 2 をこのマシンで実際に動かしてピークを測ってはいません。
他プロジェクトの速度やメモリは各プロジェクトの自己申告です。

## 0. 追記 (2026-09-22)

この文書は **「モデル全体を RAM に載せる」前提**で書かれています。その前提は
[計算律速と SSD ストリーミングの実測](../measurements/2026-09-22-m3pro-compute-vs-ssd.md)
で覆りました。要点:

- M3 Pro の行列積は 5.1〜5.9 TFLOPS、内蔵 SSD は 4.6〜5.2 GB/s。Krea 2 の 1 ステップは
  計算 約 29 秒に対して重みの読み出しは q8 で 2.8 秒。**I/O は計算に隠れる。**
- 28 ブロック分 3.88GB を毎周ディスクから流し直す実証で、常駐は 0.17GB、
  速度は計算律速のまま。
- したがって §5 の「16GB / 18GB 機で問題になる点」と §8-1 の「段階的なロード」は、
  どちらも全ブロック常駐を前提にした中間解です。
  [現在の計画](../../../.cursor/plans/2026-09-22-krea2-block-streaming.md)を参照。

**量子化ビット数について(前提の訂正)**: 本文の §3 や §5 は q4 / q3 を現実的な
選択肢として扱っていますが、**これは採りません**。8bit 未満は画像としては破綻しない
ものの、描き込みが減り表現が変わることが CUDA 側の検証で分かっています。
**q8 が要件**であり、「載らないから 4bit に落とす」は検証に値しない回避策です。
載らないなら載らないと数字つきで結論します。

以下の本文は、ウェイトの入手元(§2)、量子化と省メモリ実装の読み解き(§3, §4)、
M5 の Neural Accelerator(§6)、他プロジェクト(§7)については現在も有効です。

## 1. 結論

- ウェイトは Hugging Face の `krea/Krea-2-Turbo` から `snapshot_download` で自動取得される。
- 16GB 機向けの Krea 2 専用の省メモリ処理は無い。共通の `-q` 量子化と `--low-ram` を使うだけ。
- 最大の障害はテキストエンコーダ(TE)。量子化されず bf16 のまま約 8GB 載り、
  しかもトランスフォーマーと同時に実体化される。q4 でもロード時ピークは約 16GB になる。
- TE を破棄した後のデノイズ中は q4 で約 7.4GB + アクティベーションなので、そこは収まる。
- つまり直すべきはロード順序であり、量子化ビット数をこれ以上下げることではない。
- M5 の Neural Accelerator は(MLX 経由では)float の行列積を速くするだけ。
  CUDA の INT8 ConvRot に相当する「ハードに合わせた整数形式」は存在しない。
  形式は画質とメモリで選べばよい。

## 2. ウェイトの入手元

| 項目 | 内容 |
|---|---|
| Turbo(推論用) | `krea/Krea-2-Turbo`。`krea-2` / `krea2` エイリアス |
| Raw(学習用) | `krea/Krea-2-Raw`。`krea-2-raw` / `krea2-raw` エイリアス |
| 定義箇所 | `src/mflux/models/common/config/model_config.py:253` |
| 保存先 | `~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/` |
| 認証 | リポジトリは `gated: auto`。HF で規約に同意し `hf auth login` が必要 |

解決順序は `PathResolution.resolve`
(`src/mflux/models/common/resolution/path_resolution.py:26`)が決める。

1. ローカルパスが存在すればそれを使う。
2. HF キャッシュに完全なスナップショットがあればそれを使う。
3. どちらも無ければ `snapshot_download(repo_id, allow_patterns=...)` で取得する。

取得するファイルは `Krea2WeightDefinition.get_download_patterns`
(`src/mflux/models/krea2/weights/krea2_weight_definition.py:111`)が決める。

| ファイル | サイズ | 備考 |
|---|---|---|
| `turbo.safetensors`(ルート直下) | 26.28 GB | トランスフォーマー。bf16、約 13.1B パラメータ |
| `text_encoder/model.safetensors` | 8.88 GB | Qwen3-VL-4B。vision tower を含む |
| `vae/diffusion_pytorch_model.safetensors` | 0.51 GB | Qwen-Image VAE |
| `tokenizer/**` | 小 | |
| 合計 | 約 35.7 GB | README の「~33 GB」は GiB 換算とほぼ一致 |

補足:

- Turbo のリポジトリには diffusers 形式の `transformer/` シャード(約 26GB)もあるが、
  重複なので取得しない。
- Raw は単一ファイル版が無いので `transformer/` シャードを取得する。
- ロード時は `_select_transformer_variant` がディスク上のレイアウトを見て、
  ネイティブ形式と diffusers 形式のキーマッピングを切り替える。
- TE の vision tower のキーは `strip_te_prefix` が `None` を返して捨てる。
  `mx.load` は遅延ロードなので、捨てたテンソルは読まれない。
- TE のタップ層は `(2, 5, ..., 35)` の 12 層。最終層(36 層目)と最終 norm は使われていない。

## 3. 量子化の実装

### オンザフライ量子化(`-q 3/4/5/6/8`)

流れは `Krea2Initializer.init`(`src/mflux/models/krea2/krea2_initializer.py:20`)にある。

1. `WeightLoader.load` が `mx.load` で bf16 を遅延ロードする。
2. `WeightApplier.apply_and_quantize` が `model.update` の後に
   `nn.quantize(group_size=64, bits=...)` をかける
   (`src/mflux/models/common/weights/loading/weight_applier.py:133`)。
3. `mx.eval(model)` で全コンポーネントを一括して実体化する。

毎回 26GB の元ファイルを読んで量子化し直すので、16GB 機では起動が非常に重い。

### 量子化の対象

- `quantization_predicate` は、入力次元が 64 で割り切れる Linear だけを量子化する。
  txtfusion の `Linear(12→1)` などは対象外。
- TE は `skip_quantization=True`(`krea2_weight_definition.py:52`)。
  コメントは「量子化すると conditioning が劣化する」。
- VAE は量子化されない。
- `nn.quantize` に `mode` を渡していないので、使えるのは affine のみ。
  MLX 0.32 が持つ `mxfp4` / `mxfp8` / `nvfp4` は `-q` からは選べない。

### 事前量子化(`mflux-save`)

```sh
mflux-save --model krea-2 -q 4 --path ~/models/krea2-q4
```

- `src/mflux/models/common/cli/save.py:55` で Krea 2 に対応している。
- 保存物には `quantization_level` のメタデータが付く。次回は `--model /path` で指定すると
  量子化済みの形のまま読み込まれ、`-q` は無視される。
- TE は bf16 のまま保存されるので、q4 でもディスク上は約 16GB。

### トランスフォーマーのサイズ概算

group 64 の affine は 1 重みあたり +0.5bit(scale と bias)になる。

| bits | サイズ |
|---|---|
| bf16 | 26.3 GB |
| q8 | 約 14.0 GB |
| q6 | 約 10.7 GB |
| q5 | 約 9.0 GB |
| q4 | 約 7.4 GB |
| q3 | 約 5.8 GB |

## 4. メモリ節約の実装(すべて共通コード)

- **TE の自動破棄**
  - `--low-ram` なしでも、1 シードならエンコード直後に `text_encoder=None` にして
    `gc.collect()` と `mx.clear_cache()` を実行する
    (`src/mflux/callbacks/callback_manager.py:101`、
    `src/mflux/callbacks/instances/memory_saver.py:77`)。
  - 複数シードでも `prompt_cache` に埋め込みがあれば破棄する。
- **`--low-ram`**
  - MLX キャッシュを 1GB に制限する。
  - VAE のタイルデコードを有効にする(#735 で Krea 2 にも対応)。
  - 1 シードならループ後にトランスフォーマーも破棄してからデコードする。
- **個別オプション**: `--vae-tiling`、`--vae-tile-size`、`--mlx-cache-limit-gb`。

## 5. 16GB / 18GB 機で問題になる点

`Krea2Initializer.init` は 3 コンポーネントをすべて構築し、`mx.eval(model)` で一括して
実体化する。エンコード完了までは次の合計が同時に載る。

| 構成 | ロード時ピークの概算 |
|---|---|
| q8 | 14.0 + 8 + 0.5 ≈ 22 GB |
| q4 | 7.4 + 8 + 0.5 ≈ 16 GB |
| q3 | 5.8 + 8 + 0.5 ≈ 14 GB |

- GPU が使えるメモリの目安(推奨ワーキングセット)は、このマシン(18GB)で
  `mx.device_info()` の実測値が約 15.0GB(搭載メモリの約 78%)。
  16GB 機の値は未確認。古い報告では約 11GB、同じ比率なら約 12.5GB になる。
  対象機で `mx.device_info()["max_recommended_working_set_size"]` を見て確かめること。
- q4 でも、ロードからエンコードまでの間はスワップに頼ることになる。
- デノイズ中は約 7.4GB + アクティベーション。1024² なら 18GB 機で収まり、
  16GB 機はぎりぎりの見込み。
- オンザフライ量子化の `mx.eval` 中に bf16 の元テンソルがどれだけ同時に残るかは未計測。

現状のコードでの最も楽な運用:

```sh
mflux-save --model krea-2 -q 4 --path ~/models/krea2-q4   # 1 回だけ。遅く、スワップする
mflux-generate-krea2 --model ~/models/krea2-q4 --low-ram --prompt "..." --steps 8
```

## 6. M5 の Neural Accelerator と量子化形式

MLX 0.32.0 に同梱のカーネルヘッダ
(`.venv/lib/python3.14/site-packages/mlx/include/mlx/backend/metal/kernels/`)を読んで確認した。
M5 実機での計測はしていない。

- **行列積は float のみ**
  - `steel/gemm/nax.h:401` が Metal 4 のテンソル演算 `mpp::tensor_ops::matmul2d` を呼ぶ。
  - 量子化版(`quantized_nax.h:960` 付近)は、パックされた重みをブロック単位で
    float(fp16 / bf16)に展開し、`tile_matmad_nax` で掛け、結果を float で積算する。
- **W8A8 の整数行列積は無い**
  - アクティベーションは常に float。重みだけ量子化し、掛ける直前に戻す方式。
  - ConvRot の回転は、アクティベーションまで INT8 にして整数行列積に載せるための前処理。
    Mac ではその速度面の利点が無く、オンラインの Hadamard 変換のコストだけが残る。
- **fp 系も同じ構造**
  - `fp_quantized_nax.h`(mxfp4 / mxfp8 / nvfp4 用)も、重みを bfloat に展開してから
    同じ float 行列積に渡す。FP8 / FP4 のネイティブ演算があるわけではない。

形式の選び方:

- DiT のデノイズは計算量が律速なので、Neural Accelerator の効果はどの形式でも乗る。
- ビット数は速度よりもメモリと画質に効く。
- メモリが足りるなら affine q8(group 64)が無難。
- 16〜18GB 機は 4bit。一般にはグループの細かい nvfp4(group 16)や mxfp4(group 32)が
  affine q4(group 64)より画質で有利と言われるが、Krea 2 では未検証。
- q3 / q5 / q6 はパックが 8bit 境界に揃わず、展開がやや重い。
- Neural Accelerator 経路は MLX がデバイスを見て自動で選ぶ。mflux 側の対応は不要。
  M3 Pro ではこの経路は使われない。

## 7. 他プロジェクトの状況(2026-09 時点、検索ベース)

| プロジェクト | 実装 | Krea 2 対応 | 16GB | 備考 |
|---|---|---|---|---|
| Draw Things | 独自 Swift / Metal | 公式対応(v1.20260716.0)。8-bit / 6-bit / 6-bit S / 8-bit S | ○ | M5 対応が最も進んでいる。「M5 で M4 比最大 4.6 倍」と公表。コードからの制御は限定的 |
| stable-diffusion.cpp | ggml + Metal | 公式対応(`docs/krea2.md`) | ○ | 本体 Q8_0 + TE の Q4_K_M が推奨。TE を量子化するのが標準。Neural Accelerator の利用は未確認 |
| ComfyUI(PyTorch MPS) | PyTorch | bf16 と fp8 が動く | × | bf16 は 48GB 推奨。fp8 はメモリが減るだけで速度は同じ |
| LiuTianjie/krea2formac | PyTorch MPS + GGUF Q4_0 | 対応 | ○ | M1 16GB でピーク約 8GB。1024² で約 12 分と遅い |
| mflux | MLX | 対応 | △ | MLX ベースの Krea 2 実装は見つけた範囲でこれだけ |

使い分け:

- すぐ使いたいだけなら Draw Things。
- Python / MLX で制御したい、LoRA や自動化を組みたい、画質を自分で評価したいなら mflux。

## 8. mflux で必要な作業(効果の大きい順)

1. **段階的なロード**
   - TE をロードしてエンコードし、破棄してからトランスフォーマーをロードする。
   - 画質は変わらず、q4 のピークが約 16GB から約 8GB に下がる。16GB 機では必須。
   - 変更箇所の中心は `Krea2Initializer.init` と `Krea2.generate_image`。
     複数シードや `prompt_cache` との整合、`mflux-save` の経路に注意する。
2. **q4 の事前保存を前提にした運用にする**
   - 毎回 26GB を読んで量子化し直すのは 16GB 機では非現実的。
3. **TE の量子化をオプション化する**
   - 現在は `skip_quantization=True` で固定。
   - stable-diffusion.cpp は Q4_K_M の TE を標準にしているので、q8(約 4.3GB)は
     おそらく実用になる。画質確認が必要。
4. **`-q` に `mode` を通す**
   - mxfp4 / nvfp4 を試せるようにし、affine q4 と比較する。

## 9. 画質確認の方法

- 同一シード・同一プロンプトで、bf16 または q8 を基準にする。
- 比較対象は q4 affine、q4 nvfp4、TE q8。
- 参照画像の比較テスト(`tests/image_generation/`)の仕組みを流用できる。
  参照画像そのものは、明示的に頼まれない限り更新しない(リポジトリのルール)。
- 基準画像の生成には bf16 か q8 が載るマシンが必要。この 18GB 機では q8 はスワップ前提。
- 軽い指標として、デノイズ前の TE 埋め込みのコサイン類似度を併用する。
  TE の量子化の影響は、画像を生成しなくてもここで先に見られる。

## 10. 参考リンク

- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw)
- [Draw Things: Krea 2 / Ideogram 4 対応の告知](https://x.com/drawthingsapp/status/2078875679878488496)
- [Metal FlashAttention v2.5 w/ Neural Accelerators](https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural)
- [stable-diffusion.cpp docs/krea2.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/krea2.md)
- [Bambushu/krea2-turbo-mac](https://github.com/Bambushu/krea2-turbo-mac)
- [LiuTianjie/krea2formac](https://github.com/LiuTianjie/krea2formac)
- [Abiray/Krea-2-Turbo-GGUF](https://huggingface.co/Abiray/Krea-2-Turbo-GGUF)
