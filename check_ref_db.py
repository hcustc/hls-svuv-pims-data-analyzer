import pandas as pd

df = pd.read_excel(r'D:\TRAE test\物种数据库.xlsx', header=None)
print("文件形状: {}".format(df.shape))
print()

print("=== 前15行详细信息 ===")
for i in range(15):
    print("\n--- 第{}行 ---".format(i))
    row = df.iloc[i, :12]
    for col in range(min(12, len(row))):
        val = row[col]
        print("  列{}: {}".format(col, val))
