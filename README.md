# FIELDNOTE · 模型效果测试台

使用固定案例、固定 seed 比较同一模型家族的不同 checkpoint。支持 Anima V7 文生图、单图编辑、双图参考；提供数据导入、GPU 后台任务、进度、HTML / PDF 报告和离线 ZIP。

## 快速启动

控制台需要 Python ≥3.10；GPU 环境按外部运行时的要求准备。已验证的 Anima 环境为 Linux、Python 3.11、PyTorch 2.8 / CUDA 12.8。安装不会自动安装或替换 PyTorch。

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# AutoDL 下载 HF 数据前开启实例提供的学术加速：
source /etc/network_turbo
python tools/start_server.py
```

浏览器打开 `http://127.0.0.1:8765`。服务器仅监听 loopback；远程使用 SSH 隧道：

```bash
ssh -N -L 8765:127.0.0.1:8765 -p YOUR_PORT root@YOUR_HOST
```

`/etc/network_turbo` 为 AutoDL 实例文件，其他主机无需执行。后台导入继承服务启动时的网络环境。已有服务需更新网络环境时，在开启加速的 shell 中运行 `python tools/start_server.py --restart`（Linux）。

## 数据输入 UI

1. 展开「数据输入 → 导入新数据」，选择来源与格式。
2. 填写数据集 ID、抽样数量和抽样种子。受限 HF 仓库的令牌只用于本次导入，经子进程 stdin 传递，不写入配置、日志或报告。
3. 导入完成后预览输入顺序、目标图、原始指令及采样尺寸。
4. 在「实验配置」填入 `configs/experiment.example.json`，修改模型和运行时路径。
5. 勾选数据集，指定案例数与推理 seed，点击「冻结并填入实验配置」。每种任务最多选一个数据集。
6. 保存配置、检查资源、创建实验；先跑 1 张，再启动完整实验。

| 来源 | 内容 |
|---|---|
| HF 完整 TAR | 指定仓库、revision 和分片；留空文件名时选最小 TAR。完整下载并校验大小及 HF LFS SHA256 后抽样。 |
| 服务器 / 上传 TAR | 使用服务器路径，或上传 ≤8 GiB 的 TAR。角色格式另需配套 JSONL 清单。 |
| JSONL + 图片目录 | 自定义文本字段、输入字段顺序、目标字段和服务器图片根目录；目标图可选。 |

JSONL 每行必须有唯一 `id`。图片路径相对声明的根目录，不允许越界：

```json
{"id":"sample-1","prompt":"原始编辑指令","ref1":"images/a.png","ref2":"images/b.png","target":"images/target.png"}
```

### 两个完整分片的可复现导入

```bash
python tools/import_dataset.py configs/import-character.example.json --ask-token
python tools/import_dataset.py configs/import-background.example.json --ask-token
```

- 角色：`LAXMAYDAY/Anime_Character_Transfer_and_Reference_Dataset_3600plus`，`data/train-00003-of-00004.tar`，500 组双参考图 + 目标图。
- 背景：`LAXMAYDAY/anime_background_edit_data_webdataset`，`train-00000-of-00010.tar`，500 对源图 + 目标图。

示例锁定 revision 与抽样种子 `20260915`。按 SHA256 排序选取完整样本，排除缺失目标图、invalid 标记、空文本、损坏图片和重复配对；不足指定数量会失败。原始指令不改写。下载中断保留 `.partial`，重新提交同一分片可继续下载。

`state/downloads/` 保存完整分片，`state/datasets/<id>/` 保存清单、图像、预览及校验值。相同图像按内容去重，因此文件数可能少于「样本数 × 每组图片数」。

```bash
python tools/audit_data_library.py --state state --output state/data-audit.json
```

重新验证所有样本的 ID、文本、输入数量、图像尺寸、字节校验值及完整 TAR。

## Edit 输出尺寸

Edit 默认使用 `resolution_policy: "target_aspect"`。优先取目标图宽高比；没有目标图时取输入 1。目标图只有尺寸信息用于创建任务，像素不会传给推理后端。

- 像素预算默认为 `target_pixels: 1048576`（1024²，约 1 MP），可配置。
- 在 16 像素对齐的候选宽高里兼顾比例与像素数量，输出不超过预算，不统一裁成正方形。
- 例如 1920×1080 → 1360×768；1024×1535 → 832×1248。
- 实际尺寸在创建任务时冻结，同一案例的所有 seed / checkpoint 使用相同尺寸。
- T2I 使用配置的固定宽高。明确需要复现旧固定尺寸 Edit 时可指定 `resolution_policy: "fixed"`。

请求和结果记录 `geometry` 与最终 `profile`，报告显示实际参数。此前按固定 1024×1024 创建的实验保持原始锁定条件；使用新规则时应创建新实验。

## 模型资源与推理

源码包不包含模型、数据、已生成结果或外部运行时。准备 V7 checkpoint、Qwen3 0.6B 文本编码器、Qwen Image VAE、配套 `ComfyUI-Anima-Native-Context` 运行时目录，再修改实验配置。

Anima 来源：[anima-native-context-v7](https://github.com/Yidhar/anima-native-context-v7)。平台校验 V7 权重结构和运行时代码摘要，不自动回退 V2。训练权重仅缺发布说明字段时，`prepare-v7` 可生成独立兼容副本，原 tensor 字节保持一致。

```bash
python -m eval_platform.cli prepare-v7 SOURCE.safetensors DERIVED.safetensors RUNTIME_DIR
python -m eval_platform.cli freeze configs/bindings.example.json my-suite-v1
python -m eval_platform.cli preflight my-experiment.json
python -m eval_platform.cli create my-experiment.json my-run
python -m eval_platform.cli start my-run --limit 1
python -m eval_platform.cli start my-run
python -m eval_platform.cli status my-run
python -m eval_platform.cli cancel my-run
python -m eval_platform.cli start my-run --retry
```

套件支持每类 1–5000 案例、1–50 个唯一 seed，默认 5×5。计划张数为案例总数 × seed 数：单任务 500×5 为 2500 张，两任务合计 5000 张。导入或冻结不会自动启动 GPU 推理。

关闭网页不会停止后台任务。GPU 进程锁保证互斥；恢复时验证已完成图片并保留有效结果。追加 checkpoint 要求相同案例、seed、编码器、VAE、生成代码与环境：

```bash
python -m eval_platform.cli append my-run next-epoch.model.json
python -m eval_platform.cli report my-run --pdf
python -m eval_platform.cli report my-run --export
```

Krea2 T2I 调用官方 `sampling.sample`，CPU 协议测试已实现，真实权重的 GPU 实测仍取决于准备的资源。独立环境要求 Python ≥3.12、PyTorch ≥2.9；见 `configs/krea2.model.example.json`。Krea2 Edit 等待运行时接入，不生成占位图片。

## 测试与打包

```bash
python -m unittest discover -s tests -v
# 提供固定官方源码后，同时运行 Krea2 sampler CPU 对照测试：
KREA2_SOURCE_DIR=/path/to/krea2-official python -m unittest discover -s tests -v
python tools/build_source_release.py
```

源码 ZIP 输出到 `dist/`，采用明确文件清单并生成 `SOURCE_MANIFEST.json` 与 SHA256。包含 Python 包、网页资源、测试、通用示例、文档和中文字体；排除状态数据库、凭据、服务器私有配置、数据、模型、运行时副本和缓存。

以后上传 GitHub 时，解压源码 ZIP，在解压目录初始化 Git，再推送到自己的仓库。本工具不会创建或推送远程仓库。第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原始项目代码尚未指定开源许可证。
