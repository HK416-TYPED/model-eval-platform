# 上传 GitHub

本目录即仓库根目录，包含 v0.3.0 源码、网页资源、中文字体及授权说明、通用配置示例和测试。无需上传 ZIP 本身：解压后将本目录的内容放到仓库中。

## 首次上传

在 GitHub 创建空仓库后，在本目录执行（替换仓库地址）：

```bash
git init
git add .
git commit -m "Release model evaluation platform v0.3.0"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

若已有仓库，将这些文件复制进已有工作区并检查差异，然后正常提交；不要覆盖已有 `.git`。不包含 Git 历史或远程地址。

## 安装与验收

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m eval_platform.cli --state state serve
```

CPU 测试不要求 GPU；Krea2 官方采样器对照测试在缺少固定运行时或 PyTorch 时会跳过。模型推理所需资源见 README。

## 公开只读入口

```bash
python -m eval_platform.cli --state /path/to/state serve --read-only --host 0.0.0.0 --port 6008
```

写操作通过 loopback 管理入口完成，公网入口只能查看和下载已有报告。新增生成结果后，应在管理侧重新生成报告：

```bash
python -m eval_platform.cli --state /path/to/state report RUN_ID --pdf
```

## 包内容与校验

`SOURCE_MANIFEST.json` 记录每个源码文件的大小和 SHA256；ZIP 旁的 `.sha256` 文件用于验证压缩包。不要将 `state/`、模型、数据、日志、令牌或服务器私有配置加入仓库。

原始源码未指定开源许可证；上传不会自动授予开源授权。字体与外部运行时的授权说明见 `THIRD_PARTY_NOTICES.md`。
