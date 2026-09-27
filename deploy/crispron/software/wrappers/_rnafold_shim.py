#!/usr/bin/env python3
"""RNAfold 兼容 shim：用 ViennaRNA 的 Python API 复现 `RNAfold --noPS` 的输出。

为什么需要它
------------
CRISPRoff 通过 `subprocess` 调用 **RNAfold 可执行文件**（`get_rnafold_eng()`），
只取 stdout 最后一个 token 并去掉括号，作为 gRNA spacer 的自折叠自由能
（`spacer_self_fold`）。PyPI 的 `ViennaRNA` 轮子只提供 Python API（`import RNA`），
不包含命令行可执行文件，因此这里用**同一版本（2.6.4，与 CRISPRon 的
environment.yml 一致）**的 `RNA.fold()` 复现同一数值。

数值等价性依据
--------------
* `RNA.fold(seq)` 与 `RNAfold` CLI 走同一个 ViennaRNA 库、同一套默认参数
  （Turner 2004），`RNAfold` 默认就是 MFE 折叠；
* 本 shim 的输出被 CRISPRoff 解析的方式与真实 RNAfold 完全相同
  （最后 token = `(mfe)`）；
* 已用 CRISPRon 官方 `bin/test.sh` 的 golden 输出验证（见 audit 报告）。

用法：仅供 CRISPRoff 内部调用，不建议直接使用。
"""
import sys
import RNA


def _read_fasta(text):
    recs, rid, buf = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if rid is not None:
                recs.append((rid, "".join(buf)))
            rid, buf = line[1:].strip(), []
        else:
            buf.append(line)
    if rid is not None:
        recs.append((rid, "".join(buf)))
    return recs


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv or "-h" in argv or "--help" in argv:
        print(f"RNAfold (Python-API shim) ViennaRNA {getattr(RNA, '__version__', '2.6.4')}")
        return 0

    text = sys.stdin.read()
    recs = _read_fasta(text)
    if not recs:
        print("shim: 未从 stdin 读到 FASTA 记录", file=sys.stderr)
        return 1

    out = []
    for rid, seq in recs:
        seq = seq.upper().replace("T", "U")
        if not seq:
            continue
        structure, mfe = RNA.fold(seq)
        # CRISPRoff 取最后 token 并去括号 -> 必须是 "(mfe)"
        out.append(f">{rid}")
        out.append(seq)
        out.append(f"{structure} ({mfe:.2f})")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
