# Environment records

环境安装命令、版本清单、源码哈希和验证日志写在本目录。共享 `gearnet` 环境只读；新环境使用项目专属名称。

2026-09-24 用户授权的 Linux CPU 数据审计使用项目本地 `.venv-audit`：

```bash
python -m venv --system-site-packages .venv-audit
.venv-audit/bin/python -m pip install pandas==2.3.3
PYTHONPATH=src .venv-audit/bin/python -m unittest discover -s tests -p test_ppi_audit.py -v
```

实际运行版本：Python 3.13.13、NumPy 2.4.6、SciPy 1.18.0、LMDB 2.2.0、pandas 2.3.3。该 venv 继承已有系统包，仅增装缺少的依赖，未修改共享基础环境。完整版本记录为 `pip-freeze-local-audit-20260924.txt`。集群模型环境没有变更，也未将工作站 venv 上传覆盖集群 Conda 环境。
