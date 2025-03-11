import re
# 在脚本开头添加
# import matplotlib.font_manager as fm
#fm._rebuild()  # 重建字体缓存
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import os
import sys

def parse_log_file(log_content):
    """
    解析日志内容，提取每个模型的指标
    
    Args:
        log_content: 日志文件的内容字符串
    
    Returns:
        list: 包含每个模型指标字典的列表
    """
    # 使用正则表达式分割不同模型的数据块
    model_blocks = re.split(r'\n(?=[\w\-\.]+\.gguf)', log_content)
    
    models_data = []
    
    for block in model_blocks:
        if not block.strip():
            continue
            
        model_data = {}
        
        # 提取模型名称
        model_name_match = re.search(r'^([\w\-\.]+\.gguf)', block, re.MULTILINE)
        if model_name_match:
            model_data['model_name'] = model_name_match.group(1)
            
            # 从模型名称中提取基本模型名称和量化版本
            name_parts = model_name_match.group(1).split('-')
            if len(name_parts) >= 2:
                # 提取基本模型名称 (不包括量化后缀)
                model_data['base_model'] = '-'.join(name_parts[:-1])
                
                # 提取量化版本
                quant_match = re.search(r'Q\d+.*?(?=\.gguf)', name_parts[-1])
                if quant_match:
                    model_data['quantization'] = quant_match.group(0)
                else:
                    model_data['quantization'] = name_parts[-1].split('.')[0]
        
        # 提取模型类型
        model_type_match = re.search(r'model type = (.+)', block)
        if model_type_match:
            model_data['model_type'] = model_type_match.group(1)
        
        # 提取模型格式类型
        model_ftype_match = re.search(r'model ftype = (.+)', block)
        if model_ftype_match:
            model_data['model_ftype'] = model_ftype_match.group(1)
        
        # 提取模型参数
        model_params_match = re.search(r'model params = ([\d\.]+) (\w)', block)
        if model_params_match:
            value = float(model_params_match.group(1))
            unit = model_params_match.group(2)
            if unit == 'B':
                model_data['model_params'] = value * 1e9
            elif unit == 'M':
                model_data['model_params'] = value * 1e6
            elif unit == 'K':
                model_data['model_params'] = value * 1e3
            else:
                model_data['model_params'] = value
        
        # 提取模型大小
        model_size_match = re.search(r'model size = ([\d\.]+) (\w+)', block)
        if model_size_match:
            value = float(model_size_match.group(1))
            unit = model_size_match.group(2)
            if unit == 'GiB':
                model_data['model_size_gb'] = value
            elif unit == 'MiB':
                model_data['model_size_gb'] = value / 1024.0
            else:
                model_data['model_size_gb'] = value
        
        # 提取 BPW (Bits Per Weight)
        bpw_match = re.search(r'$$([\d\.]+) BPW$$', block)
        if bpw_match:
            model_data['bpw'] = float(bpw_match.group(1))
        
        # 提取 PPL (Perplexity)
        ppl_match = re.search(r'Final estimate: PPL = ([\d\.]+) \+/- ([\d\.]+)', block)
        if ppl_match:
            model_data['ppl'] = float(ppl_match.group(1))
            model_data['ppl_error'] = float(ppl_match.group(2))
        
        # 提取加载时间
        load_time_match = re.search(r'load time = \s*([\d\.]+) ms', block)
        if load_time_match:
            model_data['load_time_ms'] = float(load_time_match.group(1))
        
        # 提取 prompt 评估时间
        prompt_eval_match = re.search(r'prompt eval time = \s*([\d\.]+) ms / \s*(\d+) tokens $$\s*([\d\.]+) ms per token,\s*([\d\.]+) tokens per second$$', block)
        if prompt_eval_match:
            model_data['prompt_eval_time_ms'] = float(prompt_eval_match.group(1))
            model_data['prompt_tokens'] = int(prompt_eval_match.group(2))
            model_data['ms_per_token'] = float(prompt_eval_match.group(3))
            model_data['tokens_per_second'] = float(prompt_eval_match.group(4))
        else:
            # 调试信息
            print(f"警告: 无法从以下文本块中提取 prompt eval time 信息:")
            print("-" * 40)
            print(block[:200] + "..." if len(block) > 200 else block)  # 只打印前 200 个字符
            print("-" * 40)
        
        # 提取总时间
        total_time_match = re.search(r'total time = \s*([\d\.]+) ms / \s*(\d+) tokens', block)
        if total_time_match:
            model_data['total_time_ms'] = float(total_time_match.group(1))
            model_data['total_tokens'] = int(total_time_match.group(2))
        
        models_data.append(model_data)
    
    return models_data

def group_models_by_base_and_quant(models_data):
    """
    将模型按基础模型和量化版本分组
    """
    grouped_data = defaultdict(list)
    
    for model in models_data:
        base_model = model.get('base_model', 'Unknown')
        grouped_data[base_model].append(model)
    
    return grouped_data

def process_log_file(log_file_path):
    """
    处理日志文件并生成可视化
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(log_file_path):
            print(f"错误: 文件 '{log_file_path}' 不存在")
            return [], {}
        
        # 读取日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 解析日志内容
        models_data = parse_log_file(log_content)
        
        # 打印提取的数据
        print(f"共提取到 {len(models_data)} 个模型的信息:")
        for model in models_data:
            print(f"\n{model.get('model_name', 'Unknown')} 的指标:")
            for key, value in model.items():
                print(f"  {key}: {value}")
        
        # 转换为 DataFrame 并打印列名
        df = pd.DataFrame(models_data)
        print("\nDataFrame 的列名:", df.columns.tolist())
        
        if df.empty:
            print("警告: 未能从日志文件中提取有效数据")
            return [], {}
        
        # 将模型按基础模型分组
        grouped_models = group_models_by_base_and_quant(models_data)
        
        # 打印分组结果
        print("\n按基础模型分组结果:")
        for base_model, models in grouped_models.items():
            print(f"\n{base_model} 系列模型:")
            for model in models:
                print(f"  - {model.get('model_name', 'Unknown')} ({model.get('quantization', 'Unknown')})")
        
        return models_data, grouped_models
    
    except Exception as e:
        print(f"处理日志文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return [], {}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator

def plot_model_size_vs_ppl(models_data, output_file='model_size_vs_ppl.png'):
    """
    绘制模型大小与PPL关系图，特别增大横轴2.0-5.0区间和纵轴7.5-25区间的显示长度
    
    Args:
        models_data: 包含模型信息的字典列表
        output_file: 输出图像文件名
    """
    # 将数据转换为DataFrame
    df = pd.DataFrame(models_data)
    
    # 检查必要的列是否存在
    required_cols = ['base_model', 'model_size_gb', 'ppl', 'quantization']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # 设置超大图表尺寸，提供极大的显示空间
    plt.figure(figsize=(36, 20))  # 极大尺寸
    plt.style.use('ggplot')
    
    # 为不同的base_model分配不同的颜色和标记
    unique_models = df['base_model'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    if len(markers) < len(unique_models):
        markers = markers * (len(unique_models) // len(markers) + 1)
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    
    # 为每个base_model绘制一条线
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        # 按model_size_gb排序
        model_df = model_df.sort_values('model_size_gb')
        
        # 绘制线条
        plt.plot(model_df['model_size_gb'], model_df['ppl'], 
                 linestyle='-', 
                 color=colors[i], 
                 linewidth=3,  # 增加线宽
                 alpha=0.7,
                 zorder=1)
        
        # 绘制散点
        scatter = plt.scatter(model_df['model_size_gb'], model_df['ppl'], 
                   marker=markers[i % len(markers)], 
                   color=colors[i], 
                   s=250,  # 增大点的大小
                   label=model,
                   edgecolor='white',
                   linewidth=2,
                   alpha=0.9,
                   zorder=2)
        
        # 使用不同位置标注量化版本，避免重叠
        label_positions = ['top', 'bottom', 'left', 'right', 'top-right', 'top-left', 'bottom-right', 'bottom-left']
        position_index = 0
        
        for j, row in model_df.iterrows():
            x, y = row['model_size_gb'], row['ppl']
            
            # 根据当前点的位置索引选择标签位置
            position = label_positions[position_index % len(label_positions)]
            position_index += 1
            
            # 增加标签偏移量，使标签与点之间的距离更大
            if position == 'top':
                xytext = (0, 25)  # 增加垂直偏移
                ha = 'center'
                va = 'bottom'
            elif position == 'bottom':
                xytext = (0, -25)  # 增加垂直偏移
                ha = 'center'
                va = 'top'
            elif position == 'left':
                xytext = (-25, 0)  # 增加水平偏移
                ha = 'right'
                va = 'center'
            elif position == 'right':
                xytext = (25, 0)  # 增加水平偏移
                ha = 'left'
                va = 'center'
            elif position == 'top-right':
                xytext = (25, 25)  # 增加对角线偏移
                ha = 'left'
                va = 'bottom'
            elif position == 'top-left':
                xytext = (-25, 25)  # 增加对角线偏移
                ha = 'right'
                va = 'bottom'
            elif position == 'bottom-right':
                xytext = (25, -25)  # 增加对角线偏移
                ha = 'left'
                va = 'top'
            elif position == 'bottom-left':
                xytext = (-25, -25)  # 增加对角线偏移
                ha = 'right'
                va = 'top'
            
            # 创建带有轮廓的文本，提高可读性
            text = plt.annotate(row['quantization'], 
                         (x, y),
                         textcoords="offset points", 
                         xytext=xytext,
                         ha=ha,
                         va=va,
                         fontsize=14,  # 增大字体
                         fontweight='bold',
                         zorder=3)
            
            # 添加文本轮廓
            text.set_path_effects([
                path_effects.Stroke(linewidth=4, foreground='white'),
                path_effects.Normal()
            ])
    
    # 设置图表标题和标签
    plt.title('Model Size vs Perplexity (PPL)', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Model Size (GiB)', fontsize=22, labelpad=7.5)
    plt.ylabel('Perplexity (PPL) - Lower is Better', fontsize=22, labelpad=7.5)
    
    # 设置网格线
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # 获取当前的轴对象
    ax = plt.gca()
    
    # 设置坐标轴范围
    plt.xlim(0, 15)  # 固定横轴范围为 0-7.5 GiB
    plt.ylim(10, 71)  # 固定纵轴范围为 10-55 PPL
    
        # 创建非均匀的横轴刻度，使 2.0-5.0 区间更宽
    x_ticks = []

    # 0-2.0 区间，每 0.5 一个刻度
    for i in range(0, 20, 5):  # 使用整数参数
        x_ticks.append(i/10)

    # 2.0-5.0 区间，每 0.2 一个刻度，增大显示长度
    for i in range(20, 50, 2):  # 使用整数参数
        x_ticks.append(i/10)

    # 5.0-15.0 区间，每 1.0 一个刻度
    for i in range(50, 151, 10):  # 使用整数参数
        x_ticks.append(i/10)

    # 设置横轴刻度
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))

    # 创建非均匀的纵轴刻度，使 15-25 区间更宽
    y_ticks = []

    # 10-15 区间，每 1.0 一个刻度
    for i in range(10, 15, 1):  # 使用整数参数
        y_ticks.append(i)

    # 15-25 区间，每 0.5 一个刻度，增大显示长度
    for i in range(150, 250, 5):  # 使用整数参数
        y_ticks.append(i/10)

    # 25-55 区间，每 5.0 一个刻度
    for i in range(25, 56, 5):  # 使用整数参数
        y_ticks.append(i)
    
    # 设置纵轴刻度
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    
    # 设置横轴刻度标签格式
    def x_formatter(x, pos):
        if 2.0 <= x <= 5.0:
            return f"{x:.1f}"  # 2.0-5.0 区间显示一位小数
        else:
            return f"{x:.1f}" if x < 10 else f"{int(x)}"
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
    
    # 设置纵轴刻度标签格式
    def y_formatter(y, pos):
        if 7.5 <= y <= 25:
            return f"{y:.1f}"  # 7.5-25 区间显示一位小数
        else:
            return f"{int(y)}"
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(y_formatter))
    
    # 设置次刻度
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # 显示次刻度，并设置它们的样式
    ax.tick_params(which='minor', length=5, width=1.5, direction='out')
    ax.tick_params(which='major', length=10, width=2, direction='out')
    ax.tick_params(axis='both', labelsize=16)  # 增大刻度标签字体
    
    # 添加图例，放在图表外部右侧
    legend = plt.legend(title='Base Model', fontsize=18, title_fontsize=20, 
                loc='center left', bbox_to_anchor=(1, 0.5),
                frameon=True, framealpha=0.9, edgecolor='gray',
                markerscale=1.5)  # 增大图例标记
    legend.get_frame().set_facecolor('white')
    
    # 添加水印
    # plt.figtext(0.5, 0.01, "Model Performance Analysis Report", ha="center", fontsize=14, 
    #             color="gray", alpha=0.5, style='italic')
    
    # 添加数据标签：显示每个模型组的最佳PPL
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        if not model_df.empty:
            best_idx = model_df['ppl'].idxmin()
            best_model = model_df.loc[best_idx]
            plt.annotate(f"Best: {best_model['quantization']} ({best_model['ppl']:.2f})",
                        (best_model['model_size_gb'], best_model['ppl']),
                        textcoords="offset points",
                        xytext=(40, -40),  # 增加偏移
                        fontsize=16,  # 增大字体
                        color=colors[i],
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors[i], alpha=0.8))
    
    # 调整布局，留出足够的空间给图例
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 创建放大区域，显示重要区域
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # 创建放大区域，显示重要区域
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # 修复：使用绝对单位而不是相对单位
    axins = inset_axes(ax, width=8, height=6, loc="lower right", 
                       bbox_to_anchor=(0.95, 0.35), bbox_transform=ax.transAxes)
    
    # 设置子图的背景颜色为浅灰色，使其与主图区分
    axins.set_facecolor('#f8f8f8')
    
    # 在子图中重新绘制数据，但只显示指定区间
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        # 过滤出指定区间内的数据点
        filtered_df = model_df[(model_df['model_size_gb'] >= 2.0) & 
                              (model_df['model_size_gb'] <= 5.0) & 
                              (model_df['ppl'] >= 15) & 
                              (model_df['ppl'] <= 25)]
        
        if not filtered_df.empty:
            # 如果数据点太多，可以考虑只绘制散点而不绘制连线
            if len(filtered_df) > 6:
                # 只绘制散点，不绘制连线
                axins.scatter(filtered_df['model_size_gb'], filtered_df['ppl'], 
                             marker=markers[i % len(markers)], 
                             color=colors[i], 
                             s=150,
                             label=model,
                             edgecolor='white',
                             linewidth=1.5,
                             alpha=1.0,
                             zorder=3)
            else:
                # 绘制线条（更细、更透明）
                axins.plot(filtered_df['model_size_gb'], filtered_df['ppl'], 
                          linestyle='--',  # 使用虚线
                          color=colors[i], 
                          linewidth=1.0,   # 更细的线条
                          alpha=0.5,       # 更透明
                          zorder=1)
                
                # 绘制散点
                axins.scatter(filtered_df['model_size_gb'], filtered_df['ppl'], 
                             marker=markers[i % len(markers)], 
                             color=colors[i], 
                             s=150,
                             label=model,
                             edgecolor='white',
                             linewidth=1.5,
                             alpha=1.0,
                             zorder=3)
            
            # 使用智能标签放置算法，避免重叠
            label_positions = []  # 存储已放置的标签位置
            
            # 为每个点添加标签
            for _, row in filtered_df.iterrows():
                x, y = row['model_size_gb'], row['ppl']
                
                # 尝试不同的位置，直到找到一个不重叠的位置
                best_position = None
                min_overlap = float('inf')
                
                # 可能的位置列表
                positions = [
                    (10, 10, 'left', 'bottom'),    # 右上
                    (10, -10, 'left', 'top'),      # 右下
                    (-10, 10, 'right', 'bottom'),  # 左上
                    (-10, -10, 'right', 'top'),    # 左下
                    (0, 15, 'center', 'bottom'),   # 上
                    (0, -15, 'center', 'top'),     # 下
                    (15, 0, 'left', 'center'),     # 右
                    (-15, 0, 'right', 'center')    # 左
                ]
                
                for dx, dy, ha, va in positions:
                    # 计算新位置
                    new_pos = (x + dx/100, y + dy/100)  # 转换为数据坐标
                    
                    # 计算与其他标签的重叠
                    overlap = sum(
                        abs(new_pos[0] - pos[0]) < 0.1 and abs(new_pos[1] - pos[1]) < 0.2
                        for pos in label_positions
                    )
                    
                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_position = (dx, dy, ha, va)
                
                # 如果所有位置都有重叠，可以跳过这个标签
                if min_overlap > 0 and len(filtered_df) > 5:
                    continue
                
                # 使用最佳位置
                dx, dy, ha, va = best_position
                label_positions.append((x + dx/100, y + dy/100))
                
                # 添加标签
                text = axins.annotate(row['quantization'], 
                                     (x, y),
                                     textcoords="offset points", 
                                     xytext=(dx, dy),
                                     ha=ha,
                                     va=va,
                                     fontsize=11,
                                     fontweight='bold')
                
                # 添加文本轮廓
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])
    
    # 设置子图的坐标轴范围
    axins.set_xlim(2.0, 5.0)
    axins.set_ylim(15, 25)
    
    # 设置子图的刻度
    axins.xaxis.set_major_locator(MultipleLocator(0.5))
    axins.yaxis.set_major_locator(MultipleLocator(1.0))
    axins.xaxis.set_minor_locator(MultipleLocator(0.1))
    axins.yaxis.set_minor_locator(MultipleLocator(0.2))
    
    # 设置子图网格
    axins.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 添加子图标题
    axins.set_title('Zoomed Region (2.0-5.0 GiB, 15-25 PPL)', fontsize=14, pad=10)
    
    # 设置子图刻度标签格式
    axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    try:
        # 添加连接线，指示子图区域，使用更细、更浅的线条
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.7", lw=1)
    except:
        print("Warning: Could not add connecting lines to inset axes")
    
    # 保存图表
    plt.savefig(output_file, dpi=400, bbox_inches='tight')  # 增加DPI
    print(f"Chart saved to '{output_file}'")
    plt.close()
    
def plot_model_size_vs_load_time(models_data, output_file='model_size_vs_load_time.png'):
    """
    绘制模型大小与加载时间关系图，特别增大横轴1.5-5.5区间和纵轴30-100区间的显示长度
    
    Args:
        models_data: 包含模型信息的字典列表
        output_file: 输出图像文件名
    """
    # 将数据转换为DataFrame
    df = pd.DataFrame(models_data)
    
    # 检查必要的列是否存在
    required_cols = ['base_model', 'model_size_gb', 'load_time_ms', 'quantization']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # 设置超大图表尺寸，提供极大的显示空间 - 增加宽度
    plt.figure(figsize=(42, 20))  # 从36增加到42
    plt.style.use('ggplot')
    
    # 为不同的base_model分配不同的颜色和标记
    unique_models = df['base_model'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    if len(markers) < len(unique_models):
        markers = markers * (len(unique_models) // len(markers) + 1)
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    
    # 为每个base_model绘制一条线
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        # 按model_size_gb排序
        model_df = model_df.sort_values('model_size_gb')
        
        # 将加载时间从毫秒转换为秒，使数值更易读
        model_df['load_time_s'] = model_df['load_time_ms'] / 1000.0
        
        # 绘制线条
        plt.plot(model_df['model_size_gb'], model_df['load_time_s'], 
                 linestyle='-', 
                 color=colors[i], 
                 linewidth=3,  # 增加线宽
                 alpha=0.7,
                 zorder=1)
        
        # 绘制散点
        scatter = plt.scatter(model_df['model_size_gb'], model_df['load_time_s'], 
                   marker=markers[i % len(markers)], 
                   color=colors[i], 
                   s=250,  # 增大点的大小
                   label=model,
                   edgecolor='white',
                   linewidth=2,
                   alpha=0.9,
                   zorder=2)
        
        # 使用不同位置标注量化版本，避免重叠
        label_positions = ['top', 'bottom', 'left', 'right', 'top-right', 'top-left', 'bottom-right', 'bottom-left']
        position_index = 0
        
        for j, row in model_df.iterrows():
            x, y = row['model_size_gb'], row['load_time_s']
            
            # 根据当前点的位置索引选择标签位置
            position = label_positions[position_index % len(label_positions)]
            position_index += 1
            
            # 增加标签偏移量，使标签与点之间的距离更大
            if position == 'top':
                xytext = (0, 25)
                ha = 'center'
                va = 'bottom'
            elif position == 'bottom':
                xytext = (0, -25)
                ha = 'center'
                va = 'top'
            elif position == 'left':
                xytext = (-25, 0)
                ha = 'right'
                va = 'center'
            elif position == 'right':
                xytext = (25, 0)
                ha = 'left'
                va = 'center'
            elif position == 'top-right':
                xytext = (25, 25)
                ha = 'left'
                va = 'bottom'
            elif position == 'top-left':
                xytext = (-25, 25)
                ha = 'right'
                va = 'bottom'
            elif position == 'bottom-right':
                xytext = (25, -25)
                ha = 'left'
                va = 'top'
            elif position == 'bottom-left':
                xytext = (-25, -25)
                ha = 'right'
                va = 'top'
            
            # 创建带有轮廓的文本，提高可读性
            text = plt.annotate(row['quantization'], 
                         (x, y),
                         textcoords="offset points", 
                         xytext=xytext,
                         ha=ha,
                         va=va,
                         fontsize=14,  # 增大字体
                         fontweight='bold',
                         zorder=3)
            
            # 添加文本轮廓
            text.set_path_effects([
                path_effects.Stroke(linewidth=4, foreground='white'),
                path_effects.Normal()
            ])
    
    # 设置图表标题和标签
    plt.title('Model Size vs Load Time', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Model Size (GiB)', fontsize=22, labelpad=15)
    plt.ylabel('Load Time (seconds)', fontsize=22, labelpad=15)
    
    # 设置网格线
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # 获取当前的轴对象
    ax = plt.gca()
    
    # 设置坐标轴范围 - 缩小横轴范围以增大1.5-5.5区间的显示比例
    plt.xlim(0, 15)  # 从15缩小到8
    
    # 计算纵轴范围，确保所有数据点都可见
    y_min = 0  # 加载时间最小为0
    if not df.empty:
        y_max = min(df['load_time_ms'].max() / 1000.0 * 1.1, 2500)  # 限制最大值为150，增大30-100区间的显示比例
        plt.ylim(y_min, y_max)
    
    # 创建非均匀的横轴刻度，特别增大1.5-5.5区间的显示长度
    x_ticks = []
    
    # 0-1.5 区间，每 0.5 一个刻度
    for i in range(0, 15, 5):
        x_ticks.append(i/10)
    
    # 1.5-5.5 区间，减少刻度数量，每 0.4 一个刻度，增大显示长度
    for i in range(15, 45, 4):  # 从2改为4，减少刻度数量
        x_ticks.append(i/10)
    
    # 5.5-8.0 区间，每 0.5 一个刻度
    for i in range(55, 81, 5):
        x_ticks.append(i/10)
    
    # 设置横轴刻度
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    
    # 创建非均匀的纵轴刻度，特别增大30-100区间的显示长度
    y_ticks = []
    
    # 0-30 区间，每 10 一个刻度
    for i in range(0, 30, 10):
        y_ticks.append(i)
    
    # 30-100 区间，减少刻度数量，每 10 一个刻度，增大显示长度
    for i in range(30, 101, 10):  # 从5改为10，减少刻度数量
        y_ticks.append(i)
    
    # 100以上区间，每 50 一个刻度
    for i in range(100, int(y_max) + 50, 50):
        if i > 100:  # 避免重复添加100
            y_ticks.append(i)
    
    # 设置纵轴刻度
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    
    # 设置横轴刻度标签格式
    def x_formatter(x, pos):
        if 1.5 <= x <= 4.5:
            return f"{x:.1f}"  # 1.5-5.5 区间显示一位小数
        else:
            return f"{x:.1f}" if x < 10 else f"{int(x)}"
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
    
    # 设置纵轴刻度格式
    def y_formatter(y, pos):
        if 30 <= y <= 100:
            return f"{int(y)}"  # 30-100 区间显示整数
        else:
            return f"{int(y)}"
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(y_formatter))
    
    # 设置次刻度 - 减少次刻度密度，使主刻度间距更明显
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))  # 从0.1改为0.2
    ax.yaxis.set_minor_locator(MultipleLocator(5))    # 从1改为5
    
    # 显示次刻度，并设置它们的样式
    ax.tick_params(which='minor', length=5, width=1.5, direction='out')
    ax.tick_params(which='major', length=10, width=2, direction='out')
    ax.tick_params(axis='both', labelsize=16)  # 增大刻度标签字体
    
    # 添加图例，放在图表外部右侧
    legend = plt.legend(title='Base Model', fontsize=18, title_fontsize=20, 
                loc='center left', bbox_to_anchor=(1, 0.5),
                frameon=True, framealpha=0.9, edgecolor='gray',
                markerscale=1.5)  # 增大图例标记
    legend.get_frame().set_facecolor('white')
    
    # 调整布局，留出足够的空间给图例，增大主图区域的水平比例
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # 从0.85改为0.88
    
    # 添加子图，放大显示关键区域
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # 创建子图，专注于横轴1.5-5.5，纵轴30-100的区间 - 显著增大子图尺寸
    # 修改bbox_to_anchor参数，将子图向右下方移动
    axins = inset_axes(ax, width=16, height=10, loc="lower right",  
                       bbox_to_anchor=(0.98, 0.05),  # 从(0.95, 0.35)改为(0.98, 0.05)，向右下方移动
                       bbox_transform=ax.transAxes)
    
    # 设置子图的背景颜色为透明
    axins.patch.set_alpha(0.0)  # 将背景设为完全透明
    
    # 在子图中重新绘制数据，但只显示指定区间
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        # 过滤出指定区间内的数据点
        filtered_df = model_df[(model_df['model_size_gb'] >= 1.5) & 
                              (model_df['model_size_gb'] <= 4.5) & 
                              (model_df['load_time_ms'] / 1000.0 >= 30) &
                              (model_df['load_time_ms'] / 1000.0 <= 100)]
        
        if not filtered_df.empty:
            # 只绘制散点，不绘制连线，避免视觉混乱
            axins.scatter(filtered_df['model_size_gb'], filtered_df['load_time_ms'] / 1000.0, 
                         marker=markers[i % len(markers)], 
                         color=colors[i], 
                         s=200,  # 进一步增大点的大小
                         label=model,
                         edgecolor='white',
                         linewidth=1.5,
                         alpha=1.0,
                         zorder=3)
            
            # 使用智能标签放置算法，避免重叠
            label_positions = []  # 存储已放置的标签位置
            
            # 为每个点添加标签
            for _, row in filtered_df.iterrows():
                x, y = row['model_size_gb'], row['load_time_ms'] / 1000.0
                
                # 尝试不同的位置，直到找到一个不重叠的位置
                best_position = None
                min_overlap = float('inf')
                
                # 可能的位置列表
                positions = [
                    (10, 10, 'left', 'bottom'),    # 右上
                    (10, -10, 'left', 'top'),      # 右下
                    (-10, 10, 'right', 'bottom'),  # 左上
                    (-10, -10, 'right', 'top'),    # 左下
                    (0, 15, 'center', 'bottom'),   # 上
                    (0, -15, 'center', 'top'),     # 下
                    (15, 0, 'left', 'center'),     # 右
                    (-15, 0, 'right', 'center')    # 左
                ]
                
                for dx, dy, ha, va in positions:
                    # 计算新位置
                    new_pos = (x + dx/100, y + dy/5)  # 转换为数据坐标
                    
                    # 计算与其他标签的重叠
                    overlap = sum(
                        abs(new_pos[0] - pos[0]) < 0.1 and abs(new_pos[1] - pos[1]) < 2
                        for pos in label_positions
                    )
                    
                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_position = (dx, dy, ha, va)
                
                # 如果所有位置都有重叠，可以跳过这个标签
                if min_overlap > 0 and len(filtered_df) > 5:
                    continue
                
                # 使用最佳位置
                dx, dy, ha, va = best_position
                label_positions.append((x + dx/100, y + dy/5))
                
                # 添加标签
                text = axins.annotate(row['quantization'], 
                                     (x, y),
                                     textcoords="offset points", 
                                     xytext=(dx, dy),
                                     ha=ha,
                                     va=va,
                                     fontsize=13,  # 增大字体
                                     fontweight='bold')
                
                # 添加文本轮廓
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])
    
    # 设置子图的坐标轴范围
    axins.set_xlim(1.5, 4.5)
    axins.set_ylim(30, 100)
    
    # 设置子图的刻度 - 显著减少刻度数量，使每个刻度间距更大
    axins.xaxis.set_major_locator(MultipleLocator(1.0))  # 从0.5改为1.0
    axins.yaxis.set_major_locator(MultipleLocator(20))   # 从10改为20
    
    # 设置次级刻度，保持适当的间距
    axins.xaxis.set_minor_locator(MultipleLocator(0.2))  # 保持0.2
    axins.yaxis.set_minor_locator(MultipleLocator(5))    # 从2改为5
    
    # 设置子图网格
    axins.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 添加子图标题
    axins.set_title('Zoomed Region (1.5-4.5 GiB, 30-100s)', fontsize=16, pad=10)  # 增大字体
    
    # 设置子图刻度标签格式
    axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axins.tick_params(axis='both', labelsize=14)  # 增大刻度标签字体
    
    try:
        # 添加连接线，指示子图区域，使用更细、更浅的线条
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.7", lw=1)
    except:
        print("Warning: Could not add connecting lines to inset axes")
    
    # 保存图表
    plt.savefig(output_file, dpi=400, bbox_inches='tight')  # 增加DPI
    print(f"Chart saved to '{output_file}'")
    plt.close()


def plot_model_size_vs_token_time(models_data, output_file='model_size_vs_token_time.png'):
    """
    绘制模型大小与每个token的处理时间关系图，特别增大横轴2.0-5.0区间的刻度显示距离
    
    Args:
        models_data: 包含模型信息的字典列表
        output_file: 输出图像文件名
    """
    # 将数据转换为DataFrame
    df = pd.DataFrame(models_data)
    
    # 检查必要的列是否存在
    required_cols = ['base_model', 'model_size_gb', 'total_time_ms', 'total_tokens', 'quantization']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # 计算每个token的平均处理时间
    df['ms_per_token'] = df['total_time_ms'] / df['total_tokens']
    
    # 设置超大图表尺寸，提供极大的显示空间 - 增加宽度，提供更多横向空间
    plt.figure(figsize=(42, 20))  # 增加宽度从36到42
    plt.style.use('ggplot')
    
    # 为不同的base_model分配不同的颜色和标记
    unique_models = df['base_model'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    if len(markers) < len(unique_models):
        markers = markers * (len(unique_models) // len(markers) + 1)
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    
    # 为每个base_model绘制一条线
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        # 按model_size_gb排序
        model_df = model_df.sort_values('model_size_gb')
        
        # 绘制线条
        plt.plot(model_df['model_size_gb'], model_df['ms_per_token'], 
                 linestyle='-', 
                 color=colors[i], 
                 linewidth=3,  # 增加线宽
                 alpha=0.7,
                 zorder=1)
        
        # 绘制散点
        scatter = plt.scatter(model_df['model_size_gb'], model_df['ms_per_token'], 
                   marker=markers[i % len(markers)], 
                   color=colors[i], 
                   s=250,  # 增大点的大小
                   label=model,
                   edgecolor='white',
                   linewidth=2,
                   alpha=0.9,
                   zorder=2)
        
        # 使用不同位置标注量化版本，避免重叠
        label_positions = ['top', 'bottom', 'left', 'right', 'top-right', 'top-left', 'bottom-right', 'bottom-left']
        position_index = 0
        
        for j, row in model_df.iterrows():
            x, y = row['model_size_gb'], row['ms_per_token']
            
            # 根据当前点的位置索引选择标签位置
            position = label_positions[position_index % len(label_positions)]
            position_index += 1
            
            # 增加标签偏移量，使标签与点之间的距离更大
            if position == 'top':
                xytext = (0, 25)
                ha = 'center'
                va = 'bottom'
            elif position == 'bottom':
                xytext = (0, -25)
                ha = 'center'
                va = 'top'
            elif position == 'left':
                xytext = (-25, 0)
                ha = 'right'
                va = 'center'
            elif position == 'right':
                xytext = (25, 0)
                ha = 'left'
                va = 'center'
            elif position == 'top-right':
                xytext = (25, 25)
                ha = 'left'
                va = 'bottom'
            elif position == 'top-left':
                xytext = (-25, 25)
                ha = 'right'
                va = 'bottom'
            elif position == 'bottom-right':
                xytext = (25, -25)
                ha = 'left'
                va = 'top'
            elif position == 'bottom-left':
                xytext = (-25, -25)
                ha = 'right'
                va = 'top'
            
            # 创建带有轮廓的文本，提高可读性
            text = plt.annotate(row['quantization'], 
                         (x, y),
                         textcoords="offset points", 
                         xytext=xytext,
                         ha=ha,
                         va=va,
                         fontsize=14,  # 增大字体
                         fontweight='bold',
                         zorder=3)
            
            # 添加文本轮廓
            text.set_path_effects([
                path_effects.Stroke(linewidth=4, foreground='white'),
                path_effects.Normal()
            ])
    
    # 设置图表标题和标签
    plt.title('Model Size vs Token Processing Time', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Model Size (GiB)', fontsize=22, labelpad=15)
    plt.ylabel('Time per Token (ms)', fontsize=22, labelpad=15)
    
    # 设置网格线
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # 获取当前的轴对象
    ax = plt.gca()
    
    # 设置坐标轴范围 - 缩小横轴范围，增大2.0-5.0区间的显示比例
    plt.xlim(0, 15.0)  # 减小横轴范围从7.5到6.0，使得2.0-5.0区间占据更大比例
    
    # 计算纵轴范围，确保所有数据点都可见
    y_min = 0  # 处理时间最小为0
    if not df.empty:
        y_max = df['ms_per_token'].max() * 1.1  # 增加10%的边距
        plt.ylim(y_min, y_max)
    
    # 创建非均匀的横轴刻度，特别增大2.0-5.0区间的显示长度
    x_ticks = []
    
    # 0-2.0 区间，每 0.5 一个刻度
    for i in range(0, 20, 5):
        x_ticks.append(i/10)
    
    # 2.0-5.0 区间，减少刻度数量，使每个刻度间距更大
    for i in range(20, 50, 3):  # 从每0.1一个刻度改为每0.3一个刻度
        x_ticks.append(i/10)
    
    # 5.0-6.0 区间，每 0.5 一个刻度
    for i in range(50, 61, 5):
        x_ticks.append(i/10)
    
    # 设置横轴刻度
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    
    # 创建非均匀的纵轴刻度，特别增大0-200区间的显示长度
    y_ticks = []
    
    # 0-200 区间，每 20 一个刻度，减少刻度数量
    for i in range(0, 201, 20):
        y_ticks.append(i)
    
    # 200以上区间，每 100 一个刻度
    max_token_time = int(np.ceil(df['ms_per_token'].max()))
    if max_token_time > 200:
        for i in range(200, max_token_time + 100, 100):
            y_ticks.append(i)
    
    # 设置纵轴刻度
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    
    # 设置横轴刻度标签格式
    def x_formatter(x, pos):
        if 2.0 <= x <= 5.0:
            return f"{x:.1f}"  # 2.0-5.0 区间显示一位小数
        else:
            return f"{x:.1f}" if x < 10 else f"{int(x)}"
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
    
    # 设置纵轴刻度格式
    def y_formatter(y, pos):
        if y <= 200:
            return f"{int(y)}"  # 0-200 区间显示整数
        else:
            return f"{int(y)}"
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(y_formatter))
    
    # 设置次刻度 - 减少次刻度数量，使主刻度间距更明显
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))  # 保持0.1的次刻度
    ax.yaxis.set_minor_locator(MultipleLocator(10))  # 从5改为10，减少次刻度数量
    
    # 显示次刻度，并设置它们的样式
    ax.tick_params(which='minor', length=5, width=1.5, direction='out')
    ax.tick_params(which='major', length=10, width=2, direction='out')
    ax.tick_params(axis='both', labelsize=16)  # 增大刻度标签字体
    
    # 添加图例，放在图表外部右侧
    legend = plt.legend(title='Base Model', fontsize=18, title_fontsize=20, 
                loc='center left', bbox_to_anchor=(1, 0.5),
                frameon=True, framealpha=0.9, edgecolor='gray',
                markerscale=1.5)  # 增大图例标记
    legend.get_frame().set_facecolor('white')
    
    # 添加数据标签：显示每个模型组的最快处理时间
    # for i, (model, model_df) in enumerate(df.groupby('base_model')):
    #     if not model_df.empty:
    #         best_idx = model_df['ms_per_token'].idxmin()
    #         best_model = model_df.loc[best_idx]
    #         plt.annotate(f"Fastest: {best_model['quantization']} ({best_model['ms_per_token']:.2f}ms)",
    #                     (best_model['model_size_gb'], best_model['ms_per_token']),
    #                     textcoords="offset points",
    #                     xytext=(40, -40),  # 增加偏移
    #                     fontsize=16,  # 增大字体
    #                     color=colors[i],
    #                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors[i], alpha=0.8))
    
    # 调整布局，留出足够的空间给图例，同时增大主图区域的水平比例
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # 增加水平空间比例从0.85到0.88
    
    # 添加子图，放大显示关键区域
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # 创建子图，专注于横轴2.0-5.0，纵轴0-200的区间
    axins = inset_axes(ax, width=10, height=7, loc="lower right",  # 增加子图尺寸
                       bbox_to_anchor=(0.95, 0.35), bbox_transform=ax.transAxes)
    
    # 设置子图的背景颜色为浅灰色，使其与主图区分
    axins.set_facecolor('#f8f8f8')
    axins.patch.set_alpha(0.0)
    # 在子图中重新绘制数据，但只显示指定区间
    for i, (model, model_df) in enumerate(df.groupby('base_model')):
        # 过滤出指定区间内的数据点
        filtered_df = model_df[(model_df['model_size_gb'] >= 2.0) & 
                              (model_df['model_size_gb'] <= 5.0) & 
                              (model_df['ms_per_token'] <= 200)]
    
        if not filtered_df.empty:
            # 绘制散点
            axins.scatter(filtered_df['model_size_gb'], filtered_df['ms_per_token'], 
                         marker=markers[i % len(markers)], 
                         color=colors[i], 
                         s=180,  # 增大点的大小
                         label=model,
                         edgecolor='white',
                         linewidth=1.5,
                         alpha=1.0,
                         zorder=3)
            
            # 使用智能标签放置算法，避免重叠
            label_positions = []  # 存储已放置的标签位置
            
            # 为每个点添加标签
            for _, row in filtered_df.iterrows():
                x, y = row['model_size_gb'], row['ms_per_token']
                
                # 尝试不同的位置，直到找到一个不重叠的位置
                best_position = None
                min_overlap = float('inf')
                
                # 可能的位置列表
                positions = [
                    (10, 10, 'left', 'bottom'),    # 右上
                    (10, -10, 'left', 'top'),      # 右下
                    (-10, 10, 'right', 'bottom'),  # 左上
                    (-10, -10, 'right', 'top'),    # 左下
                    (0, 15, 'center', 'bottom'),   # 上
                    (0, -15, 'center', 'top'),     # 下
                    (15, 0, 'left', 'center'),     # 右
                    (-15, 0, 'right', 'center')    # 左
                ]
                
                for dx, dy, ha, va in positions:
                    # 计算新位置
                    new_pos = (x + dx/100, y + dy/10)  # 转换为数据坐标
                    
                    # 计算与其他标签的重叠
                    overlap = sum(
                        abs(new_pos[0] - pos[0]) < 0.1 and abs(new_pos[1] - pos[1]) < 10
                        for pos in label_positions
                    )
                    
                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_position = (dx, dy, ha, va)
                
                # 如果所有位置都有重叠，可以跳过这个标签
                if min_overlap > 0 and len(filtered_df) > 5:
                    continue
                
                # 使用最佳位置
                dx, dy, ha, va = best_position
                label_positions.append((x + dx/100, y + dy/10))
                
                # 添加标签
                text = axins.annotate(row['quantization'], 
                                     (x, y),
                                     textcoords="offset points", 
                                     xytext=(dx, dy),
                                     ha=ha,
                                     va=va,
                                     fontsize=11,
                                     fontweight='bold')
                
                # 添加文本轮廓
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])
    
    # 设置子图的坐标轴范围
    axins.set_xlim(2.0, 5.0)
    axins.set_ylim(0, 200)
    
    # 设置子图的刻度 - 减少刻度数量，使每个刻度间距更大
    axins.xaxis.set_major_locator(MultipleLocator(0.5))  # 保持0.5的主刻度
    axins.yaxis.set_major_locator(MultipleLocator(25))  # 从20改为25，减少刻度数量
    axins.xaxis.set_minor_locator(MultipleLocator(0.1))  # 保持0.1的次刻度
    axins.yaxis.set_minor_locator(MultipleLocator(5))   # 保持5的次刻度
    
    # 设置子图网格
    axins.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 添加子图标题
    axins.set_title('Zoomed Region (2.0-5.0 GiB, 0-200ms)', fontsize=14, pad=10)
    
    # 设置子图刻度标签格式
    axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    try:
        # 添加连接线，指示子图区域，使用更细、更浅的线条
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.7", lw=1)
    except:
        print("Warning: Could not add connecting lines to inset axes")
    
    # 保存图表
    plt.savefig(output_file, dpi=400, bbox_inches='tight')  # 增加DPI
    print(f"Chart saved to '{output_file}'")
    plt.close()

if __name__ == "__main__":
    # 设置 matplotlib 使用非交互式后端，避免在无 GUI 环境中出错
    plt.switch_backend('agg')
    
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
    else:
        log_file_path = "/home/wu/workspace/LLM_Distribution_Center/model_log.log"  # 默认日志文件路径
    
    print(f"正在处理日志文件: {log_file_path}")
    models_data, grouped_models = process_log_file(log_file_path)
    
    # 绘制模型大小与PPL关系图
    if models_data:
        plot_model_size_vs_ppl(models_data)
        
        # 添加新图表
        plot_model_size_vs_load_time(models_data)
        plot_model_size_vs_token_time(models_data)
    
    # 输出分析结果摘要
    if models_data:
        print("\n分析结果摘要:")
        df = pd.DataFrame(models_data)
        
        # PPL 最低的模型
        if 'ppl' in df.columns and not df['ppl'].isna().all():
            best_ppl_idx = df['ppl'].idxmin()
            best_ppl_model = df.loc[best_ppl_idx, 'model_name']
            best_ppl = df.loc[best_ppl_idx, 'ppl']
            print(f"PPL 最低的模型: {best_ppl_model} (PPL = {best_ppl:.4f})")
        
        # 推理速度最快的模型
        if 'tokens_per_second' in df.columns and not df['tokens_per_second'].isna().all():
            fastest_idx = df['tokens_per_second'].idxmax()
            fastest_model = df.loc[fastest_idx, 'model_name']
            fastest_speed = df.loc[fastest_idx, 'tokens_per_second']
            print(f"推理速度最快的模型: {fastest_model} ({fastest_speed:.2f} tokens/second)")
        
        # 加载时间最短的模型
        if 'load_time_ms' in df.columns and not df['load_time_ms'].isna().all():
            fastest_load_idx = df['load_time_ms'].idxmin()
            fastest_load_model = df.loc[fastest_load_idx, 'model_name']
            fastest_load = df.loc[fastest_load_idx, 'load_time_ms']
            print(f"加载时间最短的模型: {fastest_load_model} ({fastest_load:.2f} ms)")
        
        # 每个token处理时间最短的模型
        if 'total_time_ms' in df.columns and 'total_tokens' in df.columns:
            df['ms_per_token'] = df['total_time_ms'] / df['total_tokens']
            fastest_token_idx = df['ms_per_token'].idxmin()
            fastest_token_model = df.loc[fastest_token_idx, 'model_name']
            fastest_token_time = df.loc[fastest_token_idx, 'ms_per_token']
            print(f"每个token处理时间最短的模型: {fastest_token_model} ({fastest_token_time:.2f} ms/token)")
    else:
        print("未能从日志文件中提取有效数据，无法生成分析结果")