import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

# 设置输出文件夹
output_folder = '/home/wu/workspace/LLM_Distribution_Center/data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置绘图风格
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

# 读取CSV文件
file_path = '/home/wu/workspace/LLM_Distribution_Center/data/data.csv'  # 替换为您的文件路径
df = pd.read_csv(file_path)

# 创建任务类型映射字典
task_type_mapping = {
    0: 'TC',
    1: 'NER',
    2: 'QA',
    3: 'TL',
    4: 'SG'
}

# 添加任务类型名称列
df['task_type_name'] = df['task_type'].map(task_type_mapping)

# 1. 基本任务类型分布 - 饼图
def plot_task_type_pie():
    plt.figure(figsize=(10, 10))
    
    # 计算每种任务类型的数量
    task_counts = df['task_type_name'].value_counts()
    
    # 创建自定义颜色映射
    colors = sns.color_palette('viridis', len(task_counts))
    
    # 绘制饼图，设置楔形之间的间隔，但不使用3D效果
    wedges, texts, autotexts = plt.pie(
        task_counts, 
        labels=task_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=[0.05] * len(task_counts),  # 所有楔形稍微分离
        shadow=False,  # 不使用阴影
        textprops={'fontsize': 14},
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    
    # 设置自动文本的样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    plt.title('Distribution of Task Types', fontsize=18, fontweight='bold', pad=20)
    plt.axis('equal')  # 保持饼图为圆形
    
    # 添加图例，显示每种任务类型的具体数量
    legend_labels = [f"{label} ({count})" for label, count in zip(task_counts.index, task_counts)]
    plt.legend(wedges, legend_labels, title="Task Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_type_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 2. 任务类型分布 - 高级条形图
def plot_task_type_bar():
    plt.figure(figsize=(12, 8))
    
    # 计算每种任务类型的数量
    task_counts = df['task_type_name'].value_counts().sort_index()
    
    # 创建渐变色映射
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ['#2980b9', '#8e44ad'])
    colors = cmap(np.linspace(0, 1, len(task_counts)))
    
    # 绘制条形图
    bars = plt.bar(
        task_counts.index, 
        task_counts.values,
        color=colors,
        width=0.6,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'{int(height)}',
            ha='center', 
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.title('Count of Tasks by Type', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Task Type', fontsize=14)
    plt.ylabel('Number of Tasks', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整y轴以便在底部留出一些空间
    plt.ylim(0, max(task_counts.values) * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_type_bar.png'), dpi=300)
    plt.close()

# 3. 任务类型的令牌分布 - 箱线图
def plot_token_distribution_by_task():
    plt.figure(figsize=(14, 8))
    
    # 创建箱线图
    sns.boxplot(
        x='task_type_name', 
        y='task_token', 
        data=df,
        palette='viridis',
        linewidth=1.5,
        fliersize=5,
        notch=True  # 添加凹口以显示中位数的置信区间·
    )
    
    plt.title('Distribution of Token Counts by Task Type', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Task Type', fontsize=14)
    plt.ylabel('Token Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'token_distribution_boxplot.png'), dpi=300)
    plt.close()

# 4. 任务ID与任务类型的关系 - 散点图
def plot_task_id_vs_type():
    plt.figure(figsize=(14, 8))
    
    # 为每种任务类型创建不同的颜色
    unique_types = df['task_type'].unique()
    colors = sns.color_palette('viridis', len(unique_types))
    color_dict = {task_type: color for task_type, color in zip(unique_types, colors)}
    
    # 为每个任务类型绘制散点图
    for task_type in unique_types:
        subset = df[df['task_type'] == task_type]
        plt.scatter(
            subset.index,  # 使用数据框索引作为x轴
            subset['task_type'],
            label=task_type_mapping[task_type],
            color=color_dict[task_type],
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.title('Task Types Distribution Across Dataset', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Data Point Index', fontsize=14)
    plt.ylabel('Task Type', fontsize=14)
    plt.yticks(unique_types, [task_type_mapping[t] for t in unique_types], fontsize=12)
    
    # 添加图例
    plt.legend(title='Task Types', title_fontsize=12, fontsize=10)
    
    # 添加网格线
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_id_vs_type.png'), dpi=300)
    plt.close()

# 5. 任务类型分布 - 高级圆环图
def plot_task_type_donut():
    plt.figure(figsize=(12, 10))
    
    # 计算每种任务类型的数量
    task_counts = df['task_type_name'].value_counts()
    
    # 创建圆环图
    wedges, texts = plt.pie(
        task_counts, 
        labels=None,
        startangle=90,
        colors=sns.color_palette('viridis', len(task_counts)),
        wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2},
        counterclock=False
    )
    
    # 添加中心圆以创建圆环效果
    circle = plt.Circle((0, 0), 0.25, fc='white')
    plt.gca().add_artist(circle)
    
    # 在圆环中心添加总数
    plt.text(0, 0, f"Total\n{len(df)}", ha='center', va='center', fontsize=18, fontweight='bold')
    
    # 添加带有百分比和数量的图例
    legend_labels = [f"{label}: {count} ({count/sum(task_counts)*100:.1f}%)" 
                    for label, count in zip(task_counts.index, task_counts)]
    plt.legend(wedges, legend_labels, title="Task Types", loc="center left", 
               bbox_to_anchor=(0.9, 0, 0.5, 1), fontsize=12)
    
    plt.title('Distribution of Task Types', fontsize=18, fontweight='bold', pad=20)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_type_donut.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 6. 任务类型与参考值关系 - 小提琴图
def plot_reference_value_by_task():
    # 检查reference_value是否为数值型
    if pd.api.types.is_numeric_dtype(df['reference_value']):
        plt.figure(figsize=(14, 10))
        
        # 创建小提琴图
        sns.violinplot(
            x='task_type_name', 
            y='reference_value', 
            data=df,
            palette='viridis',
            inner='quartile',  # 显示四分位数
            linewidth=1.5,
            cut=0  # 不延伸超出数据范围
        )
        
        plt.title('Distribution of Reference Values by Task Type', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Task Type', fontsize=14)
        plt.ylabel('Reference Value', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'reference_value_violin.png'), dpi=300)
        plt.close()
    else:
        print("Reference value is not numeric, skipping violin plot.")

# 7. 任务类型分布 - 树状图
def plot_task_type_treemap():
    try:
        import squarify
        
        plt.figure(figsize=(12, 10))
        
        # 计算每种任务类型的数量
        task_counts = df['task_type_name'].value_counts()
        
        # 创建树状图
        squarify.plot(
            sizes=task_counts.values,
            label=[f"{label}\n{count}" for label, count in zip(task_counts.index, task_counts)],
            color=sns.color_palette('viridis', len(task_counts)),
            alpha=0.8,
            pad=0.02,
            text_kwargs={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'}
        )
        
        plt.axis('off')
        plt.title('Task Types Distribution (Treemap)', fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'task_type_treemap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("squarify package not installed, skipping treemap. Install with: pip install squarify")

# 8. 任务类型随时间变化 - 假设task_id代表时间顺序
def plot_task_type_timeline():
    plt.figure(figsize=(16, 8))
    
    # 对数据按task_id排序
    df_sorted = df.sort_values('task_id')
    
    # 为每种任务类型创建累积计数
    task_types = sorted(df['task_type'].unique())
    counts = {t: np.zeros(len(df_sorted)) for t in task_types}
    
    for t in task_types:
        mask = (df_sorted['task_type'] == t).astype(int).values
        counts[t] = np.cumsum(mask)
    
    # 绘制堆叠面积图
    x = range(len(df_sorted))
    bottom = np.zeros(len(df_sorted))
    
    for i, t in enumerate(task_types):
        plt.fill_between(
            x, 
            bottom, 
            bottom + counts[t],
            label=task_type_mapping[t],
            alpha=0.7,
            linewidth=1,
            edgecolor='white'
        )
        bottom += counts[t]
    
    plt.title('Cumulative Growth of Task Types', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Task ID (Sequential Order)', fontsize=14)
    plt.ylabel('Cumulative Count', fontsize=14)
    plt.legend(title='Task Types', title_fontsize=12, fontsize=10)
    
    # 添加网格线
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_type_timeline.png'), dpi=300)
    plt.close()

# 9. 任务类型与令牌数关系 - 热图
def plot_task_token_heatmap():
    # 创建token区间
    df['token_bin'] = pd.cut(df['task_token'], bins=10)
    
    # 创建交叉表
    heatmap_data = pd.crosstab(df['task_type_name'], df['token_bin'])
    
    plt.figure(figsize=(16, 10))
    
    # 绘制热图
    sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        annot=True,
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Tasks'}
    )
    
    plt.title('Task Types vs Token Count Ranges', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Token Count Range', fontsize=14)
    plt.ylabel('Task Type', fontsize=14)
    
    # 旋转x轴标签以便更好地显示
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_token_heatmap.png'), dpi=300)
    plt.close()

# 10. 任务类型比例 - 堆叠柱状图（按ID分组）
def plot_task_type_stacked_bars():
    plt.figure(figsize=(14, 8))
    
    # 将数据分成几个组
    num_groups = 10
    df['id_group'] = pd.qcut(df['task_id'], num_groups, labels=False)
    
    # 计算每个组中每种任务类型的比例
    proportions = pd.crosstab(
        df['id_group'], 
        df['task_type_name'],
        normalize='index'
    )
    
    # 绘制堆叠柱状图
    proportions.plot(
        kind='bar', 
        stacked=True,
        colormap='viridis',
        figsize=(14, 8),
        width=0.8,
        edgecolor='white',
        linewidth=1
    )
    
    plt.title('Task Type Proportions Across Dataset Segments', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Dataset Segment (by Task ID)', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Task Types', title_fontsize=12)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'task_type_stacked_bars.png'), dpi=300)
    plt.close()

# 执行所有可视化
plot_task_type_pie()
plot_task_type_bar()
#plot_token_distribution_by_task()
#plot_task_id_vs_type()
plot_task_type_donut()
#plot_reference_value_by_task()
plot_task_type_treemap()
plot_task_type_timeline()
#plot_task_token_heatmap()
#plot_task_type_stacked_bars()

print(f"所有可视化图表已保存到 '{output_folder}' 文件夹")