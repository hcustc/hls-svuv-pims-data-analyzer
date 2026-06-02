import pandas as pd

# 读取参考文件
excel_path = r"C:\Users\Administrator\Desktop\1.xlsx"
print(f"=== 读取参考文件: {excel_path} ===")

# 读取所有sheet
xls = pd.ExcelFile(excel_path)
print(f"工作表数量: {len(xls.sheet_names)}")
print(f"工作表名称: {xls.sheet_names}")

# 读取第一个sheet
df = pd.read_excel(excel_path, sheet_name=0)
print(f"\n数据尺寸: {df.shape}")
print(f"\n列名:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print(f"\n前20行数据:")
print(df.head(20).to_string())

print(f"\n数据类型:")
print(df.dtypes)

# 检查是否有重复物种
print(f"\n=== 检查重复物种 ===")
duplicates = df[df.duplicated(['物种名称'], keep=False)]
print(f"重复物种数量: {len(duplicates)}")
if len(duplicates) > 0:
    print("重复物种示例:")
    print(duplicates.head(10))
