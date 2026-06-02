import pandas as pd
import sys
from pathlib import Path

# 读取Excel文件
excel_path = r"C:\Users\Administrator\Desktop\物种数据库.xlsx"
output_path = r"C:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\excel_analysis.txt"

output = []
def log(msg):
    output.append(str(msg))
    print(msg)

log(f"=== 读取Excel文件 ===")
log(f"文件路径: {excel_path}")

# 读取所有sheet
xls = pd.ExcelFile(excel_path)
log(f"\n工作表数量: {len(xls.sheet_names)}")
log(f"工作表名称: {xls.sheet_names}")

# 读取第一个sheet
sheet_name = xls.sheet_names[0]
log(f"\n{'='*60}")
log(f"工作表: {sheet_name}")
log(f"{'='*60}")

df = pd.read_excel(excel_path, sheet_name=sheet_name)
log(f"行数: {len(df)}")
log(f"列数: {len(df.columns)}")
log(f"\n列名:")
for i, col in enumerate(df.columns):
    log(f"  {i+1}. {col}")

log(f"\n前10行数据:")
for idx, row in df.head(10).iterrows():
    log(f"Row {idx}: {dict(row)}")

# 保存到文件
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

log(f"\n\n完整分析已保存到: {output_path}")
