# 只读复核

```bash
python3.8 -B -m v12_r2.release
python3.8 -B -m v12_r2.cli backtest c4
python3.8 -B -m v12_r2.inspect main_three_gates_1 --scenario lag1_11bp
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.8 -B -m unittest discover -s v12_r2/tests -v
```

主扫描、选择、统计均已冻结；入口拒绝覆盖选择。`features.npz`、完整866条四情景`paths.npz`和独立参考矩阵保留在本目录，不需要重建旧cache。矩阵和JSON均小于100MB。

只校验明确的冻结模型/行情/证据；不会把另外获准维护的生产推送代码误判为研究修改。没有网络访问、真实账户写入或订单。
