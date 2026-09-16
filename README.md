# FIELDNOTE · Model Evaluation Platform

[English](#english) | [简体中文](#简体中文)

## English

Compare checkpoints from the same model family using fixed cases and seeds. Supports Anima V7 text-to-image, single-image editing, and dual-image reference tasks, with dataset imports, background GPU jobs, progress tracking, HTML / PDF reports, and offline ZIP exports.

Current version: **v0.3.0**. Compact HTML reports offer seed browsing, side-by-side checkpoint comparisons, search, pagination, and full-resolution image viewing. PDFs use a compact five-column layout and retain complete prompts and reproduction details in appendices. See [CHANGELOG.md](CHANGELOG.md) for release notes and [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md) for source upload instructions.

### Public read-only access

```bash
python -m eval_platform.cli --state /path/to/state serve --read-only --host 0.0.0.0 --port 6008
```

The read-only endpoint allows viewing results and downloading existing HTML / PDF reports; all write requests return 403. Keep the administration endpoint on its default loopback address and access it through an SSH tunnel. Both endpoints can share the same state directory.

HTML and PDF reports are generated separately. After updating templates or completing inference, run `python -m eval_platform.cli --state /path/to/state report RUN_ID --pdf` from the administration environment to refresh both. PDF download links include version parameters, and the read-only endpoint disables PDF caching.

### Quick start

The console requires Python ≥3.10. Prepare the GPU environment according to the external runtime requirements. The validated Anima environment uses Linux, Python 3.11, and PyTorch 2.8 / CUDA 12.8. Installing this package does not automatically install or replace PyTorch.

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# AutoDL only: enable the instance's network acceleration before HF downloads.
source /etc/network_turbo
python tools/start_server.py
```

Open `http://127.0.0.1:8765`. The server listens on loopback only; use an SSH tunnel for remote access:

```bash
ssh -N -L 8765:127.0.0.1:8765 -p YOUR_PORT root@YOUR_HOST
```

`/etc/network_turbo` is provided by AutoDL; skip that command on other hosts. Background imports inherit the server's network environment at startup. To update an existing server's network environment, run `python tools/start_server.py --restart` from the shell with acceleration enabled (Linux).

### Dataset import UI

1. Open “数据输入 → 导入新数据” (Data input → Import new data) and choose the source and format.
2. Enter a dataset ID, sample count, and sampling seed. Tokens for restricted HF repositories are used only for the current import, passed through subprocess stdin, and never written to configuration, logs, or reports.
3. After importing, preview input order, target images, original prompts, and sampling dimensions.
4. Load `configs/experiment.example.json` into “实验配置” (Experiment configuration) and update model and runtime paths.
5. Select datasets, set the case count and inference seeds, and click “冻结并填入实验配置” (Freeze and fill experiment configuration). Select at most one dataset per task type.
6. Save the configuration, check resources, and create an experiment. Run one image first, then start the full experiment.

| Source | Details |
|---|---|
| Complete HF TAR | Specify a repository, revision, and shard. If the filename is empty, the smallest TAR is selected. The complete archive is downloaded and its size and HF LFS SHA256 are verified before sampling. |
| Server / uploaded TAR | Use a server path or upload a TAR ≤8 GiB. Include a JSONL manifest or one JSON file per sample, or specify an external JSONL manifest. |
| JSONL + image directory | Configure the prompt field, ordered input fields, target field, and server image root. Target images are optional. |

#### Data format examples

Datasets are not restricted to particular repositories or image subjects. Choose the task structure by the number of inputs, then map your field names. Target images are optional and used only for comparison. Each JSONL line must have a unique `id`. Image paths are relative to the declared image root or TAR root and must not escape that root.

| Task structure | `task` | Example `input_fields` |
|---|---|---|
| Prompt only | `t2i` | `[]` |
| One reference image + prompt | `edit_single` | `["source"]` |
| Two reference images + prompt | `edit_dual` | `["source", "reference"]` |

Dual-image sample (one JSONL line):

```json
{"id":"sample-1","prompt":"Original editing instruction","source":"images/a.png","reference":"images/b.png","target":"images/target.png"}
```

Single-image sample:

```json
{"id":"sample-1","prompt":"Original editing instruction","source":"images/a.png"}
```

Prompt-only sample:

```json
{"id":"sample-1","prompt":"A mountain lake at sunrise."}
```

`prompt_field` selects the text field. The order of `input_fields` determines the image order received by the model. Set `target_field` to `"target"` to read a comparison image, or to an empty string to ignore targets. Replace field names with those in your dataset. Preview pairings before freezing the test suite; the platform does not infer image semantics from filenames alone.

#### Generic import configuration

For a TAR hosted on Hugging Face, save the following as `configs/my-import.json` and replace the repository, revision, shard path, and field mappings. The TAR can contain `metadata.jsonl` with the corresponding images, or one JSON file per sample. A `metadata.jsonl` at the HF repository root is also read automatically.

```json
{
  "dataset_id": "my-eval-data-v1",
  "source": "hf",
  "format": "tar",
  "repo": "YOUR_ORG/YOUR_DATASET",
  "revision": "YOUR_COMMIT_SHA",
  "filename": "data/train-00000.tar",
  "task": "edit_dual",
  "input_fields": ["source", "reference"],
  "prompt_field": "prompt",
  "target_field": "target",
  "count": 100,
  "selection_seed": 42
}
```

Public repositories can be imported directly. For restricted repositories, append `--ask-token` and enter the token when prompted:

```bash
python tools/import_dataset.py configs/my-import.json
```

For other sources, retain the task and field mappings and change the source parameters:

| Source | Configuration changes |
|---|---|
| Local TAR | Set `source` to `"local_tar"`, retaining `format: "tar"`. Replace `repo`, `revision`, and `filename` with `archive_path: "/path/to/data.tar"`. Use `manifest_path` for an external manifest if needed. |
| Local JSONL + images | Set both `source` and `format` to `"manifest"`. Replace HF source parameters with `manifest_path: "/path/to/metadata.jsonl"` and `root: "/path/to/data"`. See `configs/import-manifest.example.json` for a complete configuration. |

For reproducible sampling, fix the dataset version, shard, field mappings, sample count, and `selection_seed`; use a commit SHA for the HF `revision`. The platform samples in deterministic SHA256 order, validates the required inputs, text, and images, and fails if too few valid samples remain. Generic formats do not require a target image for every sample. Original prompts are preserved. Interrupted HF downloads retain `.partial` files and can resume when the same shard is submitted again.

`state/downloads/` stores complete shards. `state/datasets/<id>/` stores manifests, images, previews, and checksums. Identical images are deduplicated by content, so the file count may be lower than the sample count multiplied by the number of images per sample.

```bash
python tools/audit_data_library.py --state state --output state/data-audit.json
```

This revalidates sample IDs, text, input counts, image dimensions, byte checksums, and complete TAR archives.

### Edit output dimensions

Editing defaults to `resolution_policy: "target_aspect"`. It uses the target image's aspect ratio when available, otherwise the first input image's ratio. Only target dimensions are used to create the job; target pixels are never sent to the inference backend.

- The configurable pixel budget defaults to `target_pixels: 1048576` (1024², approximately 1 MP).
- Candidate dimensions are aligned to 16 pixels, balancing aspect ratio and pixel count without exceeding the budget or forcing every image into a square.
- Examples: 1920×1080 → 1360×768; 1024×1535 → 832×1248.
- Actual dimensions are frozen when jobs are created. All seeds and checkpoints for the same case use the same dimensions.
- T2I uses the configured fixed width and height. Use `resolution_policy: "fixed"` to reproduce earlier editing runs with fixed dimensions.

Requests and results record `geometry` and the final `profile`; reports show the actual parameters. Existing experiments created at a fixed 1024×1024 retain their original conditions. Create a new experiment to apply the new sizing policy.

### Model resources and inference

The source package does not include models, datasets, generated results, or external runtimes. Prepare a V7 checkpoint, Qwen3 0.6B text encoder, Qwen Image VAE, and the corresponding `ComfyUI-Anima-Native-Context` runtime directory, then update the experiment configuration.

Anima source: [anima-native-context-v7](https://github.com/Yidhar/anima-native-context-v7). The platform validates V7 weight structure and runtime code digests without automatically falling back to V2. If training weights only lack release metadata fields, `prepare-v7` can create a separate compatible copy while preserving the original tensor bytes.

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

Suites support 1–5000 cases per task type and 1–50 unique seeds, defaulting to 5×5. The planned image count is the total case count multiplied by the seed count: 500×5 produces 2500 images for one task, or 5000 for two tasks with those settings. Importing or freezing data does not automatically start GPU inference.

Closing the browser does not stop background jobs. A GPU process lock prevents concurrent GPU jobs; resuming validates completed images and retains valid results. Appending a checkpoint requires the same cases, seeds, encoder, VAE, generation code, and environment:

```bash
python -m eval_platform.cli append my-run next-epoch.model.json
python -m eval_platform.cli report my-run --pdf
python -m eval_platform.cli report my-run --export
```

Krea2 T2I calls the official `sampling.sample`. CPU protocol tests are implemented; GPU validation with real weights still depends on the resources you prepare. Its separate environment requires Python ≥3.12 and PyTorch ≥2.9; see `configs/krea2.model.example.json`. Krea2 Edit awaits runtime integration and does not generate placeholder images.

### Testing and packaging

```bash
python -m unittest discover -s tests -v
# With a pinned official source checkout, also run the Krea2 sampler CPU comparison:
KREA2_SOURCE_DIR=/path/to/krea2-official python -m unittest discover -s tests -v
python tools/build_source_release.py
```

The source ZIP is written to `dist/` using an explicit file allowlist, with a generated `SOURCE_MANIFEST.json` and SHA256 checksum. It includes the Python package, web assets, tests, generic examples, documentation, and a Chinese font. It excludes state databases, credentials, private server configuration, datasets, models, runtime copies, and caches.

To upload the source ZIP to a new GitHub repository, extract it, initialize Git in the extracted directory, and push to your repository. This tool does not create or push remote repositories. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices. The original project code has not yet been assigned an open-source license.

### Compatibility

Existing `character_tar` / `webdataset` API configurations remain supported. For new datasets, use the generic `tar` / `manifest` formats and independent task fields described above.

Imported historical report snapshots support browsing, manual reviews, and re-exporting. They cannot start inference or append checkpoints.

---

## 简体中文

FIELDNOTE · 模型效果测试台

使用固定案例、固定 seed 比较同一模型家族的不同 checkpoint。支持 Anima V7 文生图、单图编辑、双图参考；提供数据导入、GPU 后台任务、进度、HTML / PDF 报告和离线 ZIP。

当前版本：**v0.3.0**。紧凑 HTML 报告支持 Seed 浏览与 Checkpoint 横向对比、搜索、分页和原图查看；PDF 使用紧凑五列布局，保留完整指令与复现附录。发布说明见 [CHANGELOG.md](CHANGELOG.md)，源码上传说明见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。

### 公网只读入口

```bash
python -m eval_platform.cli --state /path/to/state serve --read-only --host 0.0.0.0 --port 6008
```

只读入口可查看结果和下载已有 HTML / PDF，所有写请求均返回 403。管理入口继续使用 loopback 默认地址，通过 SSH 隧道访问。两入口可以共用同一状态目录。

HTML 和 PDF 是独立生成的文件；更新模板或推理完成后，在管理侧执行 `python -m eval_platform.cli --state /path/to/state report RUN_ID --pdf`，同时刷新 HTML 与 PDF。PDF 下载采用版本参数，只读入口禁用 PDF 缓存。

### 快速启动

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

### 数据输入 UI

1. 展开「数据输入 → 导入新数据」，选择来源与格式。
2. 填写数据集 ID、抽样数量和抽样种子。受限 HF 仓库的令牌只用于本次导入，经子进程 stdin 传递，不写入配置、日志或报告。
3. 导入完成后预览输入顺序、目标图、原始指令及采样尺寸。
4. 在「实验配置」填入 `configs/experiment.example.json`，修改模型和运行时路径。
5. 勾选数据集，指定案例数与推理 seed，点击「冻结并填入实验配置」。每种任务最多选一个数据集。
6. 保存配置、检查资源、创建实验；先跑 1 张，再启动完整实验。

| 来源 | 内容 |
|---|---|
| HF 完整 TAR | 指定仓库、revision 和分片；留空文件名时选最小 TAR。完整下载并校验大小及 HF LFS SHA256 后抽样。 |
| 服务器 / 上传 TAR | 使用服务器路径，或上传 ≤8 GiB 的 TAR。包内提供 JSONL 清单或每个样本一个 JSON，也可指定外部 JSONL。 |
| JSONL + 图片目录 | 自定义文本字段、输入字段顺序、目标字段和服务器图片根目录；目标图可选。 |

#### 数据格式范例

数据源不限定仓库或图片内容。先按输入数量选择任务结构，再映射数据中的字段名；目标图可选，仅用于对照。JSONL 每行必须有唯一 `id`，图片路径相对声明的图片根目录或 TAR 根目录，不允许越界。

| 任务结构 | `task` | `input_fields` 示例 |
|---|---|---|
| 仅 Prompt | `t2i` | `[]` |
| 单参考图 + Prompt | `edit_single` | `["source"]` |
| 双参考图 + Prompt | `edit_dual` | `["source", "reference"]` |

双参考图样本（单行 JSONL）：

```json
{"id":"sample-1","prompt":"原始编辑指令","source":"images/a.png","reference":"images/b.png","target":"images/target.png"}
```

单参考图样本：

```json
{"id":"sample-1","prompt":"原始编辑指令","source":"images/a.png"}
```

仅 Prompt 样本：

```json
{"id":"sample-1","prompt":"A mountain lake at sunrise."}
```

`prompt_field` 指定文本字段，`input_fields` 的排列顺序就是模型接收图片的顺序。`target_field` 设为 `"target"` 时读取对照图，留空字符串则忽略目标图。字段名可以替换成自己数据中的名称。导入后先预览配对，再冻结测试套件；平台不会仅凭文件名猜测图片语义。

#### 通用导入配置

以 Hugging Face 上的 TAR 为例，将下面内容保存为 `configs/my-import.json`，替换仓库、revision、分片路径和字段映射。TAR 内可放 `metadata.jsonl` 与对应图片，或每个样本一个 JSON；HF 仓库根目录的 `metadata.jsonl` 也会自动读取。

```json
{
  "dataset_id": "my-eval-data-v1",
  "source": "hf",
  "format": "tar",
  "repo": "YOUR_ORG/YOUR_DATASET",
  "revision": "YOUR_COMMIT_SHA",
  "filename": "data/train-00000.tar",
  "task": "edit_dual",
  "input_fields": ["source", "reference"],
  "prompt_field": "prompt",
  "target_field": "target",
  "count": 100,
  "selection_seed": 42
}
```

公开仓库可直接导入；受限仓库在命令末尾加 `--ask-token`，按提示输入令牌：

```bash
python tools/import_dataset.py configs/my-import.json
```

其他来源保留任务与字段映射，按以下规则替换来源参数：

| 来源 | 配置调整 |
|---|---|
| 本地 TAR | `source` 改为 `"local_tar"`，保留 `format: "tar"`；用 `archive_path: "/path/to/data.tar"` 替换 `repo`、`revision`、`filename`，外部清单可用 `manifest_path` 指定。 |
| 本地 JSONL + 图片 | `source` 和 `format` 均改为 `"manifest"`；用 `manifest_path: "/path/to/metadata.jsonl"`、`root: "/path/to/data"` 替换 HF 来源参数。完整配置见 `configs/import-manifest.example.json`。 |

需要复现抽样时，固定数据版本、分片、字段映射、抽样数量和 `selection_seed`；建议将 HF `revision` 指定为提交 SHA。平台按确定性 SHA256 排序抽样，校验任务所需输入、文本与图片，不足指定数量时失败；通用格式不要求每条样本都有目标图。原始指令不改写。HF 下载中断保留 `.partial`，重新提交同一分片可继续下载。

`state/downloads/` 保存完整分片，`state/datasets/<id>/` 保存清单、图像、预览及校验值。相同图像按内容去重，因此文件数可能少于「样本数 × 每组图片数」。

```bash
python tools/audit_data_library.py --state state --output state/data-audit.json
```

重新验证所有样本的 ID、文本、输入数量、图像尺寸、字节校验值及完整 TAR。

### Edit 输出尺寸

Edit 默认使用 `resolution_policy: "target_aspect"`。优先取目标图宽高比；没有目标图时取输入 1。目标图只有尺寸信息用于创建任务，像素不会传给推理后端。

- 像素预算默认为 `target_pixels: 1048576`（1024²，约 1 MP），可配置。
- 在 16 像素对齐的候选宽高里兼顾比例与像素数量，输出不超过预算，不统一裁成正方形。
- 例如 1920×1080 → 1360×768；1024×1535 → 832×1248。
- 实际尺寸在创建任务时冻结，同一案例的所有 seed / checkpoint 使用相同尺寸。
- T2I 使用配置的固定宽高。明确需要复现旧固定尺寸 Edit 时可指定 `resolution_policy: "fixed"`。

请求和结果记录 `geometry` 与最终 `profile`，报告显示实际参数。此前按固定 1024×1024 创建的实验保持原始锁定条件；使用新规则时应创建新实验。

### 模型资源与推理

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

### 测试与打包

```bash
python -m unittest discover -s tests -v
# 提供固定官方源码后，同时运行 Krea2 sampler CPU 对照测试：
KREA2_SOURCE_DIR=/path/to/krea2-official python -m unittest discover -s tests -v
python tools/build_source_release.py
```

源码 ZIP 输出到 `dist/`，采用明确文件清单并生成 `SOURCE_MANIFEST.json` 与 SHA256。包含 Python 包、网页资源、测试、通用示例、文档和中文字体；排除状态数据库、凭据、服务器私有配置、数据、模型、运行时副本和缓存。

以后上传 GitHub 时，解压源码 ZIP，在解压目录初始化 Git，再推送到自己的仓库。本工具不会创建或推送远程仓库。第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原始项目代码尚未指定开源许可证。

### 兼容性

既有 `character_tar` / `webdataset` API 配置继续兼容；新数据建议使用上述通用 `tar` / `manifest` 格式与独立任务字段。

导入的历史报告快照支持浏览、人工评语和重新导出，不支持从快照启动推理或追加 checkpoint。
