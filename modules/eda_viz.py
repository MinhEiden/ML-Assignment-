"""
Module: eda_viz.py
Chứa các hàm vẽ biểu đồ EDA được phân loại bài bản theo từng nhóm đặc trưng:
1. Nhóm Biến mục tiêu (Target Variable): Bar Chart + Pie/Donut Chart
2. Nhóm Biến số / Liên tục (Numerical): Histogram (KDE) + Boxplot theo Target
3. Nhóm Biến phân loại danh nghĩa ít giá trị (Low-cardinality Nominal): Bar Chart + Pie Chart
4. Nhóm Biến phân loại thứ bậc (Ordinal Categorical): Ordered Bar Chart + Tỷ lệ Target theo cấp bậc
5. Nhóm Biến phân loại nhiều giá trị (High-cardinality Nominal): Top N Horizontal Bar Chart
6. Nhóm Phân tích dữ liệu khuyết (Missing Values & MNAR mechanism)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập phong cách hiển thị chuẩn cho báo cáo học thuật
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# 0. HÀM THỐNG KÊ TỔNG QUAN DỮ LIỆU KHUYẾT
# ==============================================================================
def get_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bảng thống kê chi tiết dữ liệu khuyết cho từng cột đặc trưng.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        'Kiểu dữ liệu': df.dtypes,
        'Số dòng khuyết': missing_count,
        'Tỷ lệ khuyết (%)': missing_pct.round(2),
        'Số giá trị duy nhất': df.nunique()
    }).sort_values(by='Số dòng khuyết', ascending=False)
    return summary


# ==============================================================================
# NHÓM 1: BIẾN MỤC TIÊU (TARGET VARIABLE)
# Biểu đồ phù hợp nhất: Bar Chart (số lượng & %) + Pie/Donut Chart (tỷ trọng)
# ==============================================================================
def plot_target_distribution(df: pd.DataFrame, target_col: str = 'target', 
                             save_path: str = 'reports/figures/group1_target_distribution.png', 
                             show: bool = True):
    """
    Trực quan hóa phân phối biến mục tiêu (Target Imbalance) qua Bar Chart và Donut Chart.
    """
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    total_samples = len(df)

    labels = ['Target 0\n(Không đổi việc)', 'Target 1\n(Muốn đổi việc)']
    colors = ['#2b5c8f', '#e26d5c']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. Bar Chart
    bars = axes[0].bar(labels, target_counts.values, color=colors, width=0.55, edgecolor='black', linewidth=1.2, alpha=0.9)
    axes[0].set_title('Số lượng mẫu theo nhãn Target', fontsize=13, fontweight='bold', pad=15)
    axes[0].set_ylabel('Số lượng mẫu (dòng)', fontsize=11)
    axes[0].set_ylim(0, max(target_counts.values) * 1.18)

    for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(target_counts.values) * 0.02),
            f'{count:,} mẫu\n({pct:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333'
        )

    # 2. Donut/Pie Chart
    wedges, texts, autotexts = axes[1].pie(
        target_counts.values,
        labels=labels,
        autopct='%1.2f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.75,
        explode=(0.04, 0.04),
        wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1.2)
    )

    for t in texts:
        t.set_fontsize(11)
        t.set_fontweight('semibold')
    for at in autotexts:
        at.set_fontsize(12)
        at.set_color('white')
        at.set_fontweight('bold')

    axes[1].text(0, 0, f'Tổng cộng\n{total_samples:,}\nmẫu', ha='center', va='center', fontsize=12, fontweight='bold', color='#444444')
    axes[1].set_title('Tỷ lệ phần trăm giữa các nhãn Target', fontsize=13, fontweight='bold', pad=15)

    fig.suptitle('NHÓM 1 - BIẾN MỤC TIÊU: PHÂN PHỐI TARGET (IMBALANCE ANALYSIS)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# NHÓM 2: BIẾN SỐ / LIÊN TỤC (NUMERICAL FEATURES)
# Áp dụng cho: city_development_index, training_hours
# Biểu đồ phù hợp nhất: Histogram (kèm KDE) + Boxplot phân tách theo Target
# ==============================================================================
def plot_numerical_feature(df: pd.DataFrame, col: str, target_col: str = 'target', 
                           save_path: str = None, show: bool = True):
    """
    Vẽ phân phối cho biến số: Histogram + KDE và Boxplot phân tách theo Target.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Histogram + KDE theo Target (multiple='dodge' để xếp các cột cạnh nhau, không bị pha trộn màu)
    sns.histplot(
        data=df, x=col, hue=target_col, 
        multiple='dodge', kde=True, 
        palette=['#2b5c8f', '#e26d5c'], 
        stat='density', common_norm=False, 
        shrink=0.85, alpha=0.85, edgecolor='black', linewidth=0.8,
        ax=axes[0]
    )
    axes[0].set_title(f'Phân phối Histogram & KDE của {col}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(col, fontsize=11)
    axes[0].set_ylabel('Mật độ (Density)', fontsize=11)

    # Chỉnh chú thích (legend) rõ ràng
    legend = axes[0].get_legend()
    if legend:
        legend.set_title('Target')
        for t, label in zip(legend.texts, ['0: Không đổi việc', '1: Muốn đổi việc']):
            t.set_text(label)

    # 2. Boxplot theo Target (phát hiện ngoại lai và so sánh trung vị)
    sns.boxplot(
        data=df, x=target_col, y=col, hue=target_col, legend=False, ax=axes[1],
        palette=['#2b5c8f', '#e26d5c'], width=0.4
    )
    axes[1].set_title(f'Biểu đồ Boxplot {col} theo Target', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Target (0: Không đổi việc, 1: Muốn đổi việc)', fontsize=11)
    axes[1].set_ylabel(col, fontsize=11)

    fig.suptitle(f'NHÓM 2 - BIẾN SỐ: {col.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# NHÓM 3: BIẾN PHÂN LOẠI DANH NGHĨA ÍT GIÁ TRỊ (LOW-CARDINALITY NOMINAL)
# Áp dụng cho: gender, relevent_experience, enrolled_university, major_discipline, company_type
# Biểu đồ phù hợp nhất: Bar Chart (số lượng đếm) + Pie Chart (tỷ trọng thị phần)
# ==============================================================================
def plot_nominal_feature(df: pd.DataFrame, col: str, target_col: str = 'target', 
                         save_path: str = None, show: bool = True):
    """
    Trực quan hóa biến danh nghĩa (Phương án A - Đồng nhất):
    - Subplot 1: Bar Chart số lượng quan sát & % cơ cấu (kèm nhóm (Missing)).
    - Subplot 2: Bar Chart tỷ lệ Target = 1 (%) kèm đường trung bình toàn tập.
    """
    temp_df = df[[col, target_col]].copy()
    temp_df[col] = temp_df[col].fillna('(Missing)')

    counts = temp_df[col].value_counts()
    percentages = temp_df[col].value_counts(normalize=True) * 100

    # Sắp xếp theo số lượng giảm dần, nhưng đưa (Missing) về cuối cùng nếu có
    order = counts.index.tolist()
    if '(Missing)' in order:
        order.remove('(Missing)')
        order.append('(Missing)')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # 1. Bar Chart: Số lượng quan sát & % cơ cấu
    bars = axes[0].bar(order, counts.reindex(order).values, color='#4575b4', edgecolor='black', linewidth=1.0, alpha=0.85, width=0.6)
    axes[0].set_title(f'Phân phối số lượng mẫu theo danh mục ({col})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Số lượng mẫu', fontsize=11)
    axes[0].set_xlabel(col, fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.18)
    axes[0].tick_params(axis='x', rotation=30)

    for bar, cat in zip(bars, order):
        count = counts[cat]
        pct = percentages[cat]
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(counts.values) * 0.02),
            f'{count:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333'
        )

    # 2. Bar Chart: Tỷ lệ Target = 1 (%)
    target_rates = temp_df.groupby(col)[target_col].mean().reindex(order) * 100
    bars2 = axes[1].bar(order, target_rates.values, color='#d73027', edgecolor='black', linewidth=1.0, alpha=0.85, width=0.6)
    axes[1].axhline(df[target_col].mean() * 100, color='black', linestyle='--', linewidth=1.2,
                    label=f'Trung bình toàn tập ({df[target_col].mean()*100:.1f}%)')
    axes[1].set_title(f'Tỷ lệ đổi việc (Target=1) theo danh mục ({col}) (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Tỷ lệ Target=1 (%)', fontsize=11)
    axes[1].set_xlabel(col, fontsize=11)
    axes[1].set_ylim(0, max(target_rates.dropna().values, default=50) * 1.25)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend()

    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.0,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    fig.suptitle(f'NHÓM 3 - BIẾN DANH NGHĨA: {col.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# NHÓM 4: BIẾN PHÂN LOẠI THỨ BẬC (ORDINAL CATEGORICAL)
# Áp dụng cho: education_level, company_size, experience, last_new_job
# Biểu đồ phù hợp nhất: Ordered Bar Chart + Tỷ lệ Target=1 theo từng cấp bậc
# ==============================================================================
def plot_ordinal_feature(df: pd.DataFrame, col: str, order: list, target_col: str = 'target', 
                         save_path: str = None, show: bool = True):
    """
    Trực quan hóa biến thứ bậc:
    - Subplot 1: Bar Chart sắp xếp đúng thứ tự cấp bậc tự nhiên (kèm nhóm Missing).
    - Subplot 2: Tỷ lệ Target=1 (%) theo cấp bậc để phát hiện xu hướng tăng/giảm.
    """
    temp_df = df[[col, target_col]].copy()
    temp_df[col] = temp_df[col].fillna('(Missing)')

    # Đảm bảo (Missing) có trong order nếu có dữ liệu khuyết
    actual_order = [cat for cat in order if cat in temp_df[col].unique()]
    if '(Missing)' in temp_df[col].unique() and '(Missing)' not in actual_order:
        actual_order.append('(Missing)')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # 1. Bar Chart sắp xếp theo thứ tự
    counts = temp_df[col].value_counts().reindex(actual_order).fillna(0)
    bars = axes[0].bar(actual_order, counts.values, color='#4575b4', edgecolor='black', alpha=0.85, width=0.6)
    axes[0].set_title(f'Phân phối số lượng theo thứ bậc ({col})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Số lượng mẫu', fontsize=11)
    axes[0].set_xlabel(f'{col} (Cấp bậc tăng dần $\\rightarrow$)', fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.15)
    axes[0].tick_params(axis='x', rotation=45 if len(actual_order) > 10 else 35)

    for bar, count in zip(bars, counts.values):
        if count > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(counts.values) * 0.02),
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=8 if len(actual_order) > 10 else 9, fontweight='bold',
                rotation=90 if len(actual_order) > 15 else 0
            )

    # 2. Tỷ lệ Target = 1 theo thứ bậc
    target_rates = temp_df.groupby(col)[target_col].mean().reindex(actual_order) * 100

    if col == 'experience':
        # Đối với experience (23 mốc), vẽ Line Chart xu hướng dốc xuống rất đẹp và trực quan
        x_indices = range(len(actual_order))
        axes[1].plot(x_indices, target_rates.values, color='#d73027', marker='o', linewidth=2.5, markersize=6, label='Tỷ lệ Target=1 (%)')
        axes[1].fill_between(x_indices, target_rates.values, color='#d73027', alpha=0.12)
        axes[1].axhline(df[target_col].mean() * 100, color='black', linestyle='--', linewidth=1.2,
                        label=f'Trung bình toàn tập ({df[target_col].mean()*100:.1f}%)')
        axes[1].set_xticks(x_indices)
        axes[1].set_xticklabels(actual_order, rotation=45)
        axes[1].set_title(f'Đường xu hướng tỷ lệ đổi việc theo số năm kinh nghiệm ({col})', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Tỷ lệ Target=1 (%)', fontsize=11)
        axes[1].set_xlabel(f'{col} (Số năm kinh nghiệm tăng dần $\\rightarrow$)', fontsize=11)
        axes[1].set_ylim(0, max(target_rates.dropna().values, default=50) * 1.25)
        axes[1].legend()

        # Hiển thị nhãn ở các mốc chính để không bị rối
        for i, val in enumerate(target_rates.values):
            if not np.isnan(val) and (i in [0, 1, 2, 5, 10, 15, 20, 21, len(actual_order) - 1]):
                axes[1].annotate(f'{val:.1f}%', (i, val), textcoords="offset points", xytext=(0, 6),
                                 ha='center', fontsize=8, fontweight='bold', color='#a50026')
    else:
        # Đối với các biến thứ bậc ngắn (company_size, education_level, last_new_job), giữ Bar Chart
        bars2 = axes[1].bar(actual_order, target_rates.values, color='#d73027', edgecolor='black', alpha=0.85, width=0.6)
        axes[1].axhline(df[target_col].mean() * 100, color='black', linestyle='--', linewidth=1.2,
                        label=f'Trung bình toàn tập ({df[target_col].mean()*100:.1f}%)')
        axes[1].set_title(f'Xu hướng tỷ lệ đổi việc (Target=1) theo cấp bậc ({col})', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Tỷ lệ Target=1 (%)', fontsize=11)
        axes[1].set_xlabel(f'{col} (Cấp bậc tăng dần $\\rightarrow$)', fontsize=11)
        axes[1].set_ylim(0, max(target_rates.dropna().values, default=50) * 1.25)
        axes[1].tick_params(axis='x', rotation=35)
        axes[1].legend()

        for bar in bars2:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1.0,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold'
                )

    fig.suptitle(f'NHÓM 4 - BIẾN THỨ BẬC: {col.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# NHÓM 5: BIẾN PHÂN LOẠI DANH NGHĨA NHIỀU GIÁ TRỊ (HIGH-CARDINALITY NOMINAL)
# Áp dụng cho: city (123 thành phố)
# Biểu đồ phù hợp nhất: Top N Horizontal Bar Chart
# ==============================================================================
def plot_high_cardinality_feature(df: pd.DataFrame, col: str = 'city', top_n: int = 15, 
                                  target_col: str = 'target', 
                                  save_path: str = 'reports/figures/group5_high_cardinality_city_top15.png', 
                                  show: bool = True):
    """
    Trực quan hóa biến danh nghĩa có nhiều giá trị (High-cardinality) bằng Horizontal Bar Chart.
    Hiển thị Top N nhóm có số lượng lớn nhất kèm tỷ lệ Target=1.
    """
    top_cats = df[col].value_counts().head(top_n)
    subset = df[df[col].isin(top_cats.index)]
    target_rates = subset.groupby(col)[target_col].mean().reindex(top_cats.index) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

    # 1. Horizontal Bar: Số lượng mẫu ở Top N
    bars = axes[0].barh(top_cats.index[::-1], top_cats.values[::-1], color='#3182bd', edgecolor='black', alpha=0.85)
    axes[0].set_title(f'Top {top_n} {col} có số lượng ứng viên đông nhất', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Số lượng mẫu', fontsize=11)
    axes[0].set_ylabel(col, fontsize=11)

    for bar in bars:
        w = bar.get_width()
        axes[0].text(w + max(top_cats.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{int(w):,}', ha='left', va='center', fontsize=9, fontweight='bold')

    # 2. Horizontal Bar: Tỷ lệ Target=1 tương ứng
    bars2 = axes[1].barh(target_rates.index[::-1], target_rates.values[::-1], color='#e6550d', edgecolor='black', alpha=0.85)
    axes[1].axvline(df[target_col].mean() * 100, color='black', linestyle='--', label=f'Trung bình ({df[target_col].mean()*100:.1f}%)')
    axes[1].set_title(f'Tỷ lệ Target=1 (%) tại Top {top_n} {col}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tỷ lệ Target=1 (%)', fontsize=11)
    axes[1].legend()

    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 1.0, bar.get_y() + bar.get_height() / 2,
                     f'{w:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

    fig.suptitle(f'NHÓM 5 - BIẾN DANH NGHĨA NHIỀU GIÁ TRỊ: {col.upper()} (TOP {top_n})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# NHÓM 6: DỮ LIỆU KHUYẾT VÀ TƯƠNG QUAN (MISSINGNESS & CORRELATION)
# Biểu đồ phù hợp: Horizontal Bar tỷ lệ khuyết + So sánh MNAR + Correlation Heatmap
# ==============================================================================
def plot_missing_values(df: pd.DataFrame, 
                        save_path: str = 'reports/figures/group6_missing_values_bar.png', 
                        show: bool = True):
    """
    Vẽ biểu đồ thanh ngang thể hiện tỷ lệ khuyết (%) và số lượng dòng khuyết ở các cột.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    missing_pct = (missing / len(df)) * 100

    if len(missing) == 0:
        print("Tập dữ liệu không có giá trị khuyết.")
        return

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars = ax.barh(missing.index, missing_pct.values, color='#d9534f', edgecolor='black', linewidth=1.0, alpha=0.85)
    ax.set_title('NHÓM 6 - DỮ LIỆU KHUYẾT: TỶ LỆ KHUYẾT THEO CỘT (%)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Tỷ lệ khuyết (%)', fontsize=11)
    ax.set_xlim(0, max(missing_pct.values) * 1.25)

    for bar, count, pct in zip(bars, missing.values, missing_pct.values):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f'{count:,} dòng ({pct:.2f}%)',
            ha='left', va='center', fontsize=10, fontweight='bold', color='#333333'
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_missingness_vs_target(df: pd.DataFrame, target_col: str = 'target', 
                               save_path: str = 'reports/figures/group6_missingness_vs_target.png', 
                               show: bool = True):
    """
    So sánh tỷ lệ Target = 1 giữa trường hợp Có dữ liệu vs Bị khuyết
    đối với tất cả các cột có missing values (chứng minh cơ chế MNAR).
    """
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        print("Không có cột nào bị khuyết.")
        return

    records = []
    for col in missing_cols:
        rate_present = df[df[col].notnull()][target_col].mean() * 100
        rate_missing = df[df[col].isnull()][target_col].mean() * 100
        records.append({
            'Feature': col,
            'Có dữ liệu': rate_present,
            'Bị khuyết': rate_missing
        })

    comp_df = pd.DataFrame(records).set_index('Feature')

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    comp_df.plot(kind='bar', ax=ax, color=['#2b5c8f', '#e26d5c'], width=0.6, edgecolor='black')
    ax.axhline(df[target_col].mean() * 100, color='gray', linestyle='--', label=f'Trung bình toàn tập ({df[target_col].mean()*100:.1f}%)')
    ax.set_title('NHÓM 6 - DỮ LIỆU KHUYẾT: SO SÁNH TỶ LỆ ĐỔI VIỆC (CÓ DỮ LIỆU VS BỊ KHUYẾT)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Tỷ lệ Target=1 (%)', fontsize=11)
    ax.set_xlabel('Cột đặc trưng', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    ax.legend()

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_matrix(df: pd.DataFrame, 
                            method: str = 'spearman',
                            save_path: str = 'reports/figures/group6_correlation_matrix.png', 
                            show: bool = True):
    """
    Vẽ ma trận tương quan Spearman Rank Correlation (7x7) cho cả biến số và biến thứ bậc.
    Biến thứ bậc được ánh xạ tạm thời sang dạng số (0, 1, 2,...) để đánh giá mối quan hệ đơn điệu với Target.
    """
    temp = df.copy()

    # Ánh xạ biến thứ bậc sang dạng số
    edu_map = {'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 4}
    size_map = {'<10': 0, '10/49': 1, '50-99': 2, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7}
    job_map = {'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5}

    def parse_exp(val):
        if pd.isnull(val): return np.nan
        if val == '<1': return 0
        if val == '>20': return 21
        try: return int(val)
        except: return np.nan

    temp['education_level_rank'] = temp['education_level'].map(edu_map)
    temp['company_size_rank'] = temp['company_size'].map(size_map)
    temp['experience_rank'] = temp['experience'].apply(parse_exp)
    temp['last_new_job_rank'] = temp['last_new_job'].map(job_map)

    cols = [
        'city_development_index', 'training_hours',
        'education_level_rank', 'company_size_rank',
        'experience_rank', 'last_new_job_rank', 'target'
    ]
    labels = [
        'City Dev Index', 'Training Hours',
        'Education Level', 'Company Size',
        'Experience', 'Last New Job', 'Target'
    ]

    corr = temp[cols].corr(method=method)
    corr.columns = labels
    corr.index = labels

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, linecolor='white', ax=ax, cbar_kws={'label': f'{method.capitalize()} Correlation'})
    ax.set_title(f'NHÓM 6 - TƯƠNG QUAN: MA TRẬN TƯƠNG QUAN {method.upper()} (7x7)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# HÀM BỔ TRỢ: KIỂM CHỨNG QUAN HỆ 1 - 1 GIỮA CITY VÀ CITY_DEVELOPMENT_INDEX
# ==============================================================================
def check_city_cdi_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm chứng quan hệ 1-1 giữa city và city_development_index.
    Chứng minh rằng mỗi thành phố chỉ có duy nhất 1 giá trị CDI,
    từ đó đưa ra quyết định loại bỏ cột city để giảm số chiều trong Pipeline.
    """
    cdi_per_city = df.groupby('city')['city_development_index'].nunique()
    is_one_to_one = (cdi_per_city.max() == 1) and (cdi_per_city.min() == 1)
    
    summary = pd.DataFrame({
        'Tổng số thành phố (unique cities)': [df['city'].nunique()],
        'Số lượng CDI tối đa trên mỗi thành phố': [cdi_per_city.max()],
        'Số lượng CDI tối thiểu trên mỗi thành phố': [cdi_per_city.min()],
        'Quan hệ 1 - 1 (Toàn vẹn)': ['CÓ (100% Phù hợp)' if is_one_to_one else 'KHÔNG']
    })
    return summary


# ==============================================================================
# NHÓM 7: KHẢO SÁT KHÔNG GIAN ĐẶC TRƯNG CHO KỸ THUẬT PCA (BÁM SÁT ĐỀ BÀI PDF)
# ==============================================================================
def plot_pca_scree_plot(df: pd.DataFrame, 
                        save_path: str = 'reports/figures/group7_pca_scree_plot.png', 
                        show: bool = True):
    """
    Khảo sát không gian đặc trưng cho kỹ thuật PCA (Mục 4.1 trang 3 đề bài PDF):
    - Mã hóa One-Hot Encoding cho toàn bộ tập dữ liệu (loại bỏ enrollee_id, city, target).
    - Chuẩn hóa dữ liệu bằng StandardScaler.
    - Vẽ Scree Plot (Cumulative Explained Variance Ratio) để xác định số Principal Components cho 90% và 95% phương sai.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Loại bỏ ID, city (do đã có CDI) và target
    X = df.drop(columns=['enrollee_id', 'city', 'target'], errors='ignore').copy()

    # Điền tạm thời missing để kiểm tra phân tích thành phần chính
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('Missing')
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    X_encoded = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    pca = PCA()
    pca.fit(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    n_90 = np.argmax(cum_var >= 90) + 1
    n_95 = np.argmax(cum_var >= 95) + 1
    total_components = len(cum_var)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # 1. Đường tích lũy phương sai
    ax.plot(range(1, total_components + 1), cum_var, marker='o', markersize=4, color='#2b5c8f', linewidth=2, label='Phương sai tích lũy (%)')
    ax.bar(range(1, total_components + 1), pca.explained_variance_ratio_ * 100, alpha=0.3, color='#4575b4', label='Phương sai từng thành phần (%)')

    # 2. Ngưỡng 90%
    ax.axhline(90, color='#e6550d', linestyle='--', linewidth=1.2, label=f'Ngưỡng 90% phương sai ({n_90} components)')
    ax.axvline(n_90, color='#e6550d', linestyle=':', linewidth=1.2)
    ax.plot(n_90, cum_var[n_90 - 1], marker='*', markersize=14, color='#e6550d')
    ax.annotate(f'90% tại k={n_90}', (n_90, cum_var[n_90 - 1]), textcoords="offset points", xytext=(-40, 10),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff2df", ec="#e6550d", lw=1), fontsize=9, fontweight='bold')

    # 3. Ngưỡng 95%
    ax.axhline(95, color='#d9534f', linestyle='--', linewidth=1.2, label=f'Ngưỡng 95% phương sai ({n_95} components)')
    ax.axvline(n_95, color='#d9534f', linestyle=':', linewidth=1.2)
    ax.plot(n_95, cum_var[n_95 - 1], marker='*', markersize=14, color='#d9534f')
    ax.annotate(f'95% tại k={n_95}', (n_95, cum_var[n_95 - 1]), textcoords="offset points", xytext=(-40, 10),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fde2e1", ec="#d9534f", lw=1), fontsize=9, fontweight='bold')

    ax.set_title('NHÓM 7 - KHẢO SÁT PCA: SCREE PLOT & TỶ LỆ PHƯƠNG SAI TÍCH LŨY', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Số lượng Thành phần chính (Principal Components - k)', fontsize=11)
    ax.set_ylabel('Tỷ lệ Phương sai giải thích tích lũy (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, total_components + 1)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"-> Kết quả PCA: Cần {n_90}/{total_components} components để giữ 90% phương sai, và {n_95}/{total_components} components cho 95% phương sai.")

