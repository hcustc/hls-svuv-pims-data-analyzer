import pandas as pd

file_path = r"D:\TRAE test\物种数据库.xlsx"
try:
    df = pd.read_excel(file_path, header=None)
    print("=== 文件形状:", df.shape)
    print("\n=== 前20行:")
    print(df.head(20))
    
    print("\n=== 尝试不同的header方式:")
    df2 = pd.read_excel(file_path, header=0)
    print(df2.head())
    print("\n列名:", df2.columns.tolist())
except Exception as e:
    print(f"读取Excel错误: {e}")
