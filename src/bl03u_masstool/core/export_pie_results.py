import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from bl03u_masstool.core.pie_analysis import species_ionization_energy_value


def export_pie_results_to_excel(pie_dialog, output_path: str | Path) -> dict:
    """
    导出PIE物种鉴定结果到Excel文件，格式符合摩尔分数处理小程序要求
    
    必需列：质量数、物种名称、电离能(eV)、匹配系数、贡献比例(%)、R²
    相同物种相同分子式只保留贡献比最高的一个
    """
    output_path = Path(output_path)
    
    results = []
    
    # 收集所有拟合结果
    if hasattr(pie_dialog, 'all_fit_results') and pie_dialog.all_fit_results:
        for mz, fit_result in pie_dialog.all_fit_results.items():
            if not fit_result.get('success'):
                continue
            
            # 获取模型中的物种列表
            model = fit_result.get('model')
            if not model:
                continue
            
            species_list = model.get('species', [])
            r_squared = fit_result.get('r_squared', 0)
            
            if species_list:
                for species_info in species_list:
                    # 使用正确的键名：species, ie, coefficient, contribution_percent
                    results.append({
                        '质量数': mz,
                        '物种名称': species_info.get('species', 'Unknown'),
                        '电离能(eV)': species_ionization_energy_value(species_info),
                        '匹配系数': species_info.get('coefficient', 0),
                        '贡献比例(%)': species_info.get('contribution_percent', 0),
                        'R²': r_squared
                    })
            else:
                results.append({
                    '质量数': mz,
                    '物种名称': 'Unknown',
                    '电离能(eV)': None,
                    '匹配系数': 0,
                    '贡献比例(%)': 0,
                    'R²': r_squared
                })
    
    if not results:
        return {
            'success': False,
            'message': '没有找到拟合结果数据',
            'count': 0
        }
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按质量数和贡献比例排序
    df = df.sort_values(['质量数', '贡献比例(%)'], ascending=[True, False])
    
    # 去重：相同质量数和物种名称只保留贡献比例最高的一个
    df = df.drop_duplicates(subset=['质量数', '物种名称'], keep='first')
    
    # 选择并排序列（与参考格式完全一致）
    export_df = df[['质量数', '物种名称', '电离能(eV)', '匹配系数', '贡献比例(%)', 'R²']].copy()
    
    # 四舍五入数值
    # Missing IE is a valid outcome of the lookup workflow; keep it as a blank
    # Excel cell instead of coercing it to the physically misleading value 0.
    export_df['电离能(eV)'] = pd.to_numeric(export_df['电离能(eV)'], errors='coerce').round(4)
    export_df['匹配系数'] = export_df['匹配系数'].round(6)
    export_df['贡献比例(%)'] = export_df['贡献比例(%)'].round(2)
    export_df['R²'] = export_df['R²'].round(6)
    
    # 确保质量数为整数
    export_df['质量数'] = export_df['质量数'].astype(int)
    
    # 导出到Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        worksheet = writer.sheets['Sheet1']
        
        # 调整列宽
        for column in worksheet.columns:
            max_length = 0
            column_name = column[0].value
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    return {
        'success': True,
        'message': f'成功导出 {len(export_df)} 条记录到 {output_path}',
        'count': len(export_df),
        'file_path': str(output_path)
    }
