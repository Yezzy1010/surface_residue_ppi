# 第三方源码

`ESM-GearNet/` 是官方仓库 https://github.com/DeepGraphLearning/ESM-GearNet 的 Git submodule，固定在 `bc90557bbf9ee2676e87b9fc84570aa126daa808`。父仓库只记录该 commit，不复制第三方 Git 历史或权重。其他第三方目录继续由 `.gitignore` 排除。

已有项目克隆恢复源码：

```bash
git submodule update --init third_party/ESM-GearNet
```

新克隆使用 `git clone --recurse-submodules`。不要使用 `git submodule update --remote` 自动追踪上游新版本。项目自有的封装、表面融合、patch 与二部图代码放在 `src/surface_residue_ppi/`，官方源码保持原样。

权重单独放在 `data/models/esm_gearnet/`。版本及来源见 `configs/dependencies/esm_gearnet.json`，操作说明见 `docs/esm_gearnet_setup_20261106.md`。
