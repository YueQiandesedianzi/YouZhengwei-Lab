# 游正伟课题组

> GitHub 仓库技术名：`YouZhengwei-Lab`（GitHub 仓库 URL 不支持纯中文名，故使用英文 slug；仓库标题与内容按中文课题组命名。）

本仓库用于存放课题组课题资料与代码传承包，**按文件夹**组织，便于后续继续追加新课题。

## 目录结构

```
YouZhengwei-Lab/
├── README.md
├── 聚氨酯课题/                 ← 动态聚氨酯机器学习传承包
│   ├── README.md
│   ├── DATA_MANIFEST.md
│   ├── chapter3_DA/
│   └── chapter4_SMILES/
└── 摩擦电比例预测课题/         ← PGDS/PGTD 摩擦电信号 → 配方比例深度学习（精简可跑）
    ├── README.md
    ├── requirements.txt
    ├── data/
    ├── teng_ml/
    └── outputs/                ← 示例：不训练即可重出的 Vpp/线性证明图
```

## 已上传

| 文件夹 | 内容 | 说明入口 |
|---|---|---|
| `聚氨酯课题/` | 硕士论文第三、四章相关代码与全链路数据（可复现） | [`聚氨酯课题/README.md`](./聚氨酯课题/README.md) |
| `摩擦电比例预测课题/` | PGDS/PGTD 双材料摩擦电波形的分类 + 回归精简代码与原始/外部数据；可重跑出图，方便写论文 | [`摩擦电比例预测课题/README.md`](./摩擦电比例预测课题/README.md) |

## 快速开始（摩擦电比例预测）

```bash
git clone https://github.com/leonfighting-py/YouZhengwei-Lab.git
cd YouZhengwei-Lab/摩擦电比例预测课题
python -m pip install -r requirements.txt

# 不训练，先出电信号趋势 / 线性证明图
python -m teng_ml.ratio_publication_story_panels
python -m teng_ml.build_linearity_proof_panel

# 完整训练 + DL 面板（需 PyTorch，耗时更长）
python -m teng_ml.ratio_dual_model --run-name my_run --max-epochs 200
```

更完整的「代码怎么读 / 怎么改图 / 论文叙事」见该文件夹内 README。

## 后续追加

把新课题文件夹放到本仓库根目录下即可，例如：

```
YouZhengwei-Lab/
├── 聚氨酯课题/
├── 摩擦电比例预测课题/
└── 新课题名称/     ← 下次上传
```
