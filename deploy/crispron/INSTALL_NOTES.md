# CRISPRon 安装记录

原始包 `package/crispron-main.zip` **未被修改**（sha256 见 `package/SHA256SUMS.txt`）。
`software/crispron-main/` 是按 CRISPRon README 的官方步骤完成的"安装实例"：

1. 解压 `crispron-main.zip` → `software/crispron-main/`
2. 将 CRISPRoff 1.1.2 的 `CRISPRspec_CRISPRoff_pipeline.py` 复制到 `bin/`
3. 将 CRISPRoff 1.1.2 的 `energy_dics.pkl` 复制到 `data/model/`

（对应 README 的 "CRISPRoff" 小节：CRISPRoff 需另行安装，并把上述两个文件
  分别放到 `bin/` 与 `data/model/` 下。CRISPRoff 原始 tar.gz 存于
  `dependencies/`，其 checksum 见 `dependencies/SHA256SUMS.txt`。）

## 环境
隔离 venv：`venv/`（python 3.10.20），版本对齐 CRISPRon 官方 `environment.yml`：
tensorflow 2.14.0 / keras 2.14.0 / biopython 1.83 / ViennaRNA 2.6.4 /
pandas 2.2.2 / scikit-learn 1.4.2 / numpy 1.26.4。

## RNAfold 说明（唯一偏离官方安装的点，已记录）
官方要求 ViennaRNA 的 `RNAfold` **可执行文件**（CRISPRoff 用 subprocess 调用）。
PyPI 的 `ViennaRNA==2.6.4` 轮子只提供 Python API，不含 CLI。因此提供
`software/wrappers/RNAfold`（+ `_rnafold_shim.py`），用**同一版本**的
`RNA.fold()` 复现同一 MFE，并按 CRISPRoff 的解析方式输出（最后 token = `(mfe)`）。

等价性由 CRISPRon 官方 `bin/test.sh` 的 golden 输出验证（须 `TEST ok`）。
