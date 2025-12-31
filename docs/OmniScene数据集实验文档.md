# OmniScene 数据集实验说明（GGN 分支规划）

## 配置概览
- **新增数据集配置**：在 `config/dataset/omniscene.yaml` 中沿用 `config/dataset/re10k.yaml` 的字段结构，把 `name` 改为 `omniscene`、`roots` 指向 `datasets/omniscene`，并将 `defaults.view_sampler` 固定成 `all`，以便在任意阶段一次性提供 6 个输入视角（key frame）和若干输出视角。`image_shape` 初始设为 `[224, 400]`，`background_color` / `cameras_are_circular` / `baseline_epsilon` / `skip_bad_shape` 等参数保持与 re10k 一致；`near`/`far` 参照 depthsplat 的经验值设置为 `0.5` / `100.`，同时在 dataclass 中补充 `train_times_per_scene`、`highres` 等布尔字段，用于控制帧采样节奏。
- **实验配置**：仿照 `config/experiment/re10k.yaml` 新增 `config/experiment/omniscene_112x200.yaml` 与 `config/experiment/omniscene_224x400.yaml`。两份文件都需要：
  - 在 `defaults` 中覆盖 `dataset: omniscene`、`model/encoder: costvolume`、`model/decoder: splatting_cuda`、`loss: [mse, lpips]`。模型/损失的具体超参直接沿用本项目 re10k 设定（`num_depth_candidates: 128`、`gaussians_per_pixel: 1`、`loss.lpips.weight: 0.05` 等），确保网络对齐。
  - 将 `data_loader.train/val/test.batch_size` 全部设置为 `1`，并继承 `config/main.yaml` 中的 `num_workers`、`persistent_workers` 等值，保证三个阶段的 batch size 完全一致。
  - 把 `trainer.max_steps` 设为 `100_001`，`trainer.val_check_interval` 设为 `0.01`。GGN 当前的 `TrainCfg` 不包含 depthsplat 的 `eval_model_every_n_val` 机制，因此文档中约定：测试流程需在训练若干步后手动开启 `mode=test`，而非依赖自动触发。
  - `dataset.image_shape` 分别写成 `[112, 200]` 与 `[224, 400]`，并与 `dataset` 覆写块中的 `train_times_per_scene=1`、`baseline_scale_bounds=false`、`make_baseline_1=false`、`near=0.5`、`far=100.` 配套。这样可复现 depthsplat 的扫描距离设定。
  - `test.eval_time_skip_steps=5`、`test.compute_scores=true` 直接沿用 re10k 实验，以便输出 PSNR/SSIM/LPIPS 与推理耗时。
- **运行指令**：README 已使用 `python -m src.main +experiment=...` 的方式，本实验也沿用此模式。例如：
  ```bash
  # 112x200 训练
  python -m src.main +experiment=omniscene_112x200 \
    data_loader.train.batch_size=1 data_loader.val.batch_size=1 data_loader.test.batch_size=1 \
    trainer.max_steps=100001 trainer.val_check_interval=0.01 \
    checkpointing.load=null \
    output_dir=outputs/omniscene-112x200

  # 112x200 测试
  python -m src.main +experiment=omniscene_112x200 mode=test \
    checkpointing.load=checkpoints/omniscene-112x200/checkpoints/epoch_0-step_100000.ckpt \
    test.save_image=true test.compute_scores=true
  ```
  224x400 版本只需切换实验名。因为 GGN 没有 `eval_model_every_n_val`，我们会在训练日志观察 `info/global_step` 达标后手动运行一次 `mode=test` 命令完成评估。

## 数据加载流程
1. **注册入口**：需要在 `src/dataset/__init__.py` 中把 `omniscene` 注册进 `DATASETS` 映射，并让 `DatasetCfg` 联合类型包含 `DatasetOmniSceneCfg`。这样 `DataModule` 才能根据 `cfg.dataset.name` 创建 OmniScene 数据集。
2. **数据类实现**：`src/dataset/dataset_omniscene.py` 以 depthsplat 的同名文件为模板，结合 GGN 的需求做如下设计：
   - 构造函数读取 `bins_train_3.2m.json` / `bins_val_3.2m.json` / `bins_dynamic_demo`，并保留 “mini-test” 策略：测试阶段默认执行 `self.bin_tokens = self.bin_tokens[0::14][:2048]`，若想跑全量再手动去除该行。
   - `__getitem__` 针对每个 bin 固定六个环视摄像头作为 `context`（key frame），再为每个摄像头取后续帧 `[1, 2]` 拼接成监督集，最后把输入帧也追加到 `target`，共 18 张图，与 depthsplat 保持一致。
   - 图像、掩码与内参加载交由 `src/dataset/utils_omniscene.py` 完成。该工具会读取 `samples_param_small/*.json`，在 resize 时同步缩放内参，并将 `fx/fy/cx/cy` 除以目标分辨率，从而生成 0~1 归一化的 K 矩阵，符合 GGN `projection.project`/`get_world_rays` 的假设。
   - 返回的 `context` / `target` dict 含有 `extrinsics`（c2w）、`intrinsics`、`image`、`near`、`far`、`index`，以及 `target["masks"]`（动态区域掩码）。由于 OmniScene loader 已输出指定分辨率，示例末尾不会再调用 `apply_crop_shim`，避免重复缩放。
3. **与 depthsplat 的差异及复用评估**：
   - depthsplat 的数据类继承 `torch.utils.data.Dataset`，GGN 的 re10k loader 则是 `IterableDataset`。GGN 的 `DataModule` 会根据 `isinstance(dataset, IterableDataset)` 决定是否打乱，因此我们也使用 `Dataset`，这样 DataLoader 能自动 shuffle，逻辑上可直接复用 depthsplat 的写法。
   - GGN 的 `DatasetCfgCommon` 暂无 `train_times_per_scene`、`highres` 字段，需要在 GGN 新建的 `DatasetOmniSceneCfg` dataclass 中补全这些参数，否则 Hydra 解析会报错。
   - 现有 re10k loader 依赖 `ViewSampler` 选取上下文/目标帧；OmniScene 的帧组合是手工定义的，因此新的数据类会接受 `view_sampler` 参数但不会调用，只保留接口一致性。这一点与 depthsplat（完全不使用 sampler）相同。
   - depthsplat 支持 `train.use_dynamic_mask`，而 GGN 的 `ModelWrapper` 目前只计算整幅图的 MSE/LPIPS。短期内掩码只随 batch 传递但不会作用于 loss，后续如需过滤动态区域，可在 `ModelWrapper` 中额外读取 `batch["target"]["masks"]`。
   - 综上，深度加载逻辑基本可以直接迁移，但仍需根据 GGN 的目录结构与 logging 方式做微调（例如使用 `Path`、避免裸 `print`）。

## 主程序调用方式
- **Hydra / Trainer 控制节奏**：GGN 的 `src/main.py` 与 depthsplat 的差别主要在于：没有 `eval_cfg` 双配置、也没有 `train.eval_model_every_n_val`。因此我们会沿用 re10k 的流程——训练命令执行 `trainer.fit`，验证频率靠 `trainer.val_check_interval=0.01` 控制；需要测试时就切换 `mode=test` 重启一次脚本。该节奏与 depthsplat README 中“先训练、再单独测试”的做法对齐，只是缺少自动触发。
- **ModelWrapper 使用方式**：GGN 的 `ModelWrapper.training_step` 读取 `batch["context"]`/`batch["target"]`，再把 decoder 输出与 `target["image"]` 计算 `LossMse`/`LossLpips`。与 depthsplat 相比：1) 暂无 `target["masks"]` 相关逻辑；2) 也没有 `train.use_dynamic_mask`、`save_video_omniscene` 等扩展回调。初版集成只需保证 RGB + 相机参数满足接口，后续若需要掩码或自定义可视化，再在 `ModelWrapper` 中添加分支。
- **命令差异总结**：
  1. depthsplat 的“训练时每 10 次 val 做一次 test”依赖 `train.eval_model_every_n_val=10`，GGN 没有该字段，所以测试需要手动运行第二条命令。
  2. GGN 的 re10k 默认启用 LPIPS（`loss: [mse, lpips]`）和 `costvolume` encoder，我们会保持这一组合，不在命令行频繁切换 small/base/large 架构，除非额外有算力要求。
  3. 三个阶段的 batch size 都设置为 1，`data_loader.*.num_workers` 沿用原值。与 depthsplat 相同，验证/测试 DataLoader 也走相同的 batch 设定，方便和自有方法对齐。

## 后续实现路线
1. 新增 `config/dataset/omniscene.yaml` 与两份 experiment 配置，确保 Hydra 可以解析 `+experiment=omniscene_*` 并创建 batch size=1 的 DataLoader。
2. 在 `src/dataset` 下引入 `dataset_omniscene.py` 与 `utils_omniscene.py`，并在 `DATASETS` 注册，以 depthsplat 代码为基础对齐接口（归一化内参、掩码、mini-test 抽样等）。
3. 根据上述配置编写训练/测试脚本，确认 `info/global_step`、`val/psnr_val`、`test/psnr` 等日志项能够正常输出；若需要掩码支持，再评估是否改动 `ModelWrapper`。
4. 通过本文档评审后，再进入真正的实现与调试阶段，代码风格将对齐 depthsplat 分支。
