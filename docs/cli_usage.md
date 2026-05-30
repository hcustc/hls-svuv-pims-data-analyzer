# BL03U MassSpectrumTool 批处理 CLI

`scripts/bl03u_cli.py` 提供不依赖图形界面的批处理入口，适合论文结果复现、远程服务器运行和批量导出。

推荐在项目根目录运行：

```bash
conda activate pyqt_env
python scripts/bl03u_cli.py --help
```

## PIE 曲线生成

```bash
python scripts/bl03u_cli.py pie tests/fixtures/C6F11O2H/PIE_Scan/400 \
  --no-recursive \
  --no-gaussian \
  --output output/exports/pie_400.csv \
  --curves-json output/exports/pie_400_curves.json
```

常用参数：

- `--manual-peak-path config/peak_integration.yaml`: 使用手动卡峰文件。
- `--target-mz 18,84`: 只导出指定 m/z。
- `--photon-mode first|none|off`: 光强归一化模式。
- `--light-source io|beam_current`: 光强来源。
- `--mass-discrimination VALUE`: 质量歧视校正因子。

## 温度扫描

```bash
python scripts/bl03u_cli.py temperature tests/fixtures/C6F11O2H/Temp_Scan/12.5eV \
  --reference-mode sum \
  --no-gaussian \
  --output output/exports/temp_12_5ev.csv \
  --curves-json output/exports/temp_12_5ev_curves.json
```

常用参数：

- `--manual-peak-path FILE`: 使用手动卡峰文件。
- `--reference-mode sum|max_temperature`: 参考谱构建方式。
- `--no-photon-normalize`: 不做光强归一化。
- `--kr-correct --kr-mz 84`: 启用 Kr 膨胀修正。

## 分子式与同位素

```bash
python scripts/bl03u_cli.py formula 'Ca(OH)2'
python scripts/bl03u_cli.py formula 'CuSO4.5H2O' --isotopes --output output/exports/cuso4_hydrate.json
python scripts/bl03u_cli.py formula '[13C]H4'
```

分子式解析支持括号分组、水合点、简单电荷尾缀和方括号同位素标签，例如 `Ca(OH)2`、`CuSO4.5H2O`、`NH4+`、`[13C]H4`。
