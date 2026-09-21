# Krea 2 Turbo 4-step 蒸留 LoRA を mflux で使う

- 調査日: 2026-09-22
- 対象: [`lvladikov/Krea2-Turbo-Distill-4step-LoRA`](https://huggingface.co/lvladikov/Krea2-Turbo-Distill-4step-LoRA)
- 対象コミット: `ada5323`

Turbo の最小ステップ数を 8 → 4 に下げる蒸留 LoRA。非ゲート、rank 64、438MB。

## 1. どのファイルを使うか

| ファイル | 使う? |
|---|---|
| `krea2_turbo_4step_rank_64_lora_comfyui.safetensors` | **これ**。ComfyUI キー名 (`diffusion_model.*`, `lora_down`/`lora_up`) |
| `krea2_turbo_4step_rank_64_lora.safetensors` | diffusers キー名 (`lora_A`/`lora_B` + alpha)。mflux は両方扱えるが未検証 |
| `_archive/checkpoints/*` | 不要。中間チェックポイント 20 本 ≈ 9GB |

`--lora-paths` には必ず `repo:ファイル名` 形式で渡す。リポジトリ ID だけを渡すと
`LoraResolution._load_repo_from_cache` が `*.safetensors` で snapshot を取り、
`_archive/` 配下まで落ちる。

```
--lora-paths 'lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors'
```

## 2. キーの一致(検証済み)

comfyui 版の全 **456 キーが mflux の `Krea2LoRAMapping` に 456/456 で一致**。
照合は重みを落とさずに safetensors のヘッダ(range リクエスト)だけで行える。

```python
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
mappings = LoRALoader._build_pattern_mappings(Krea2LoRAMapping.get_mapping())
# {block} を 0..27 で展開して、ファイルのキー集合と比較する
```

内訳は 224 個のブロック内 linear (attn q/k/v/gate/out + ff gate/up/down × 28) と
global 4 個。global 側のキーは `diffusion_model.tmlp.0` / `tmlp.2` / `tproj.1` /
`last.linear` で、mflux 側の別名定義(`krea2_lora_mapping.py:18-24`)にそのまま載る。

## 3. bake の扱い(q8 では bake してよい)

**このプロジェクトの対象は q8 なので、bake して構わない。** q8 ベースへの bake は
同じ q8 に戻るだけで、畳んだ後は LoRA 側の 0.44GB が不要になるぶん常駐が減る。
以下は 8bit 未満を使う場合の注意で、記録として残す。

### 8bit 未満で bake してはいけない理由

`LoRASaver._bake_delta_into_linear` (`lora_saver.py:165`) は、量子化ビット数が
8 未満の層に LoRA を畳むとき **q8 に再量子化する**。rank 64 のデルタが q4/q3 の
量子化ステップより小さく、そのまま畳むと丸め潰されるため。

この LoRA は 228 層 = トランスフォーマーの重い linear のほぼ全部を触るので、
8bit 未満のベースに bake すると、その 228 層がまるごと q8 相当に太る。
その場合は `--no-bake-lora` で runtime adapter のまま使うことになる(bf16 で 0.44GB 追加)。
本プロジェクトは q8 固定なので、この分岐には入らない。

## 4. サンプラーと guidance

- **`--scheduler euler`**: LoRA は euler / simple で蒸留されている。mflux の Krea 2 既定は
  `er_sde` なので明示が必要。
- **guidance は 1.0 のまま**(mflux の既定)。mflux は guidance ≠ 1.0 のときだけ
  無条件ブランチを作るので、1.0 = 単一パス = ComfyUI の cfg 1.0。diffusers 流儀の
  0.0 を入れてはいけない。
- **`--steps 4`** を明示(Krea 2 の既定は 8)。

## 5. σ スケジュールの一致条件

LoRA が学習した 4 点は σ = 1.0 / 0.905 / 0.760 / 0.513 で、これは固定 μ=1.15 に対応する。
mflux は公式 scheduler_config どおりの動的指数シフトを使うので、μ は解像度で変わる
(`linear_scheduler.py:32-39`, `mu = m·W·H/256 + b`, base/max shift 0.5/1.15,
base/max seq len 256/6400)。

| 解像度 | μ | σ |
|---|---|---|
| 1024×1024 | 0.906 | 1.0 / 0.881 / 0.712 / 0.452 |
| 1024×1536 | 1.123 | 1.0 / 0.902 / 0.755 / 0.506 |
| **1280×1280** | **1.150** | **1.0 / 0.905 / 0.760 / 0.513**(学習点と一致) |

1280² で μ が上限 1.15 に達する(W·H/256 = 6400 = `sigma_max_seq_len`)。
画質比較をするならまず 1280²、速度を見るなら 1024² でよい。
