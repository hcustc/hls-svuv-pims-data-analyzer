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

### Agent 友好证据导出

建议论文复现、批量分析和 Agent 辅助解析时同时导出分析清单、m/z 级证据对象和 Markdown 摘要报告：

```bash
python scripts/bl03u_cli.py pie tests/fixtures/C6F11O2H/PIE_Scan/400 \
  --no-recursive \
  --no-gaussian \
  --photon-mode off \
  --output output/exports/pie_400.csv \
  --curves-json output/exports/pie_400_curves.json \
  --manifest-json output/exports/pie_400_manifest.json \
  --evidence-json output/exports/pie_400_evidence.json \
  --report-md output/exports/pie_400_report.md
```

如需在导出证据时自动进行 PICS 候选拟合，增加数据库参数：

```bash
python scripts/bl03u_cli.py pie tests/fixtures/C6F11O2H/PIE_Scan/400 \
  --no-recursive \
  --no-gaussian \
  --photon-mode off \
  --database database/species_database.sqlite \
  --fit-pics \
  --output output/exports/pie.csv \
  --curves-json output/exports/pie_curves.json \
  --manifest-json output/exports/pie_manifest.json \
  --evidence-json output/exports/pie_evidence.json \
  --report-md output/exports/pie_report.md
```

输出文件含义：

- `analysis_manifest.json`：记录输入路径、分析类型、运行参数、软件环境、输出文件和数据规模。
- `evidence_objects.json`：按 m/z 组织曲线、PICS 拟合结果、候选数量、置信度和警告信息。
- `summary_report.md`：由 manifest 和 evidence 生成的可读报告，只汇总结构化证据，不额外生成未经验证的科学判断。

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

温度扫描同样支持证据导出：

```bash
python scripts/bl03u_cli.py temperature tests/fixtures/C6F11O2H/Temp_Scan/12.5eV \
  --reference-mode sum \
  --no-gaussian \
  --output output/exports/temp_12_5ev.csv \
  --curves-json output/exports/temp_12_5ev_curves.json \
  --manifest-json output/exports/temp_12_5ev_manifest.json \
  --evidence-json output/exports/temp_12_5ev_evidence.json \
  --report-md output/exports/temp_12_5ev_report.md
```

温度 evidence 会记录每个 m/z 的温度-面积曲线、曲线类型、判别原因、有效温度点数量和低置信度警告。Agent 或人工复核应优先查看 `evidence_objects.json` 与 `summary_report.md`，再决定是否需要回看原始曲线。

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
