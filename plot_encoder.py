import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端以支持无显示环境下绘图
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 尝试使用系统支持的中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def add_box(ax, x, y, w, h, text, fc, ec='#333333', lw=1.5, fontsize=10, round_pad=0.02):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={round_pad},rounding_size=6",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color='#222222')

def add_curved_arrow(ax, start, end, ctrl_offset=(0, 0), color='#333333', lw=1.4, arrowstyle='-|>', mutation_scale=12):
    # 三次贝塞尔曲线
    sx, sy = start
    ex, ey = end
    cx1, cy1 = sx + ctrl_offset[0], sy + ctrl_offset[1]
    cx2, cy2 = ex - ctrl_offset[0], ey - ctrl_offset[1]
    path = Path(
        [ (sx, sy),
          (cx1, cy1),
          (cx2, cy2),
          (ex, ey) ],
        [ Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4 ]
    )
    patch = FancyArrowPatch(
        path=path, color=color, lw=lw, arrowstyle=arrowstyle, mutation_scale=mutation_scale
    )
    ax.add_patch(patch)

def main():
    fig_w, fig_h = 4.2, 5.0  # 尺寸接近原图
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.axis('off')

    # 外层容器（圆角）
    outer_x, outer_y, outer_w, outer_h = 10, 5, 80, 140
    outer = FancyBboxPatch(
        (outer_x, outer_y), outer_w, outer_h,
        boxstyle="round,pad=0.02,rounding_size=10",
        linewidth=2, edgecolor='#777777', facecolor='#f7f7f7'
    )
    ax.add_patch(outer)

    # 左侧 N× 标注
    ax.text(6, 75, "N×", fontsize=13, rotation=90, ha='center', va='center', color='#333333')

    # 配色（尽量接近原图）
    c_addnorm = "#d9f0c7"   # 淡绿
    c_ffn     = "#cfeaf6"   # 淡蓝
    c_mha     = "#f8d8a5"   # 淡橙

    # 模块位置参数
    # 下层：多头注意力 + 添加与归一化
    mha_x, mha_y, mha_w, mha_h = 25, 30, 50, 32
    add1_x, add1_y, add1_w, add1_h = 25, 68, 50, 20

    # 上层：前馈网络 + 添加与归一化
    ffn_x, ffn_y, ffn_w, ffn_h = 25, 95, 50, 32
    add2_x, add2_y, add2_w, add2_h = 25, 133, 50, 20

    # 画模块
    add_box(ax, add1_x, add1_y, add1_w, add1_h, "添加与归一化", c_addnorm)
    add_box(ax, mha_x,  mha_y,  mha_w,  mha_h,  "多头注意力", c_mha)

    add_box(ax, add2_x, add2_y, add2_w, add2_h, "添加与归一化", c_addnorm)
    add_box(ax, ffn_x,  ffn_y,  ffn_w,  ffn_h,  "前馈网络",   c_ffn)

    # 中央竖线（数据主干）
    ax.plot([outer_x + outer_w/2, outer_x + outer_w/2], [outer_y + 10, outer_y + outer_h - 10],
            color='#333333', lw=1.6)

    backbone_x = outer_x + outer_w/2

    # 下层残差与进入 Add&Norm
    # 从主干 -> Add&Norm（下层）
    add_curved_arrow(ax, (backbone_x, mha_y + mha_h + 4), (add1_x + add1_w - 6, add1_y + add1_h/2),
                     ctrl_offset=(8, 16))
    # 从多头注意力 -> Add&Norm
    add_curved_arrow(ax, (mha_x + mha_w/2, mha_y + mha_h + 2), (add1_x + 6, add1_y + add1_h/2),
                     ctrl_offset=(-12, 18))

    # 从 Add&Norm（下层） -> 主干
    add_curved_arrow(ax, (add1_x + add1_w/2, add1_y + add1_h - 1), (backbone_x, ffn_y - 6),
                     ctrl_offset=(0, 18))

    # 上层残差与进入 Add&Norm
    # 从主干 -> Add&Norm（上层）
    add_curved_arrow(ax, (backbone_x, ffn_y + ffn_h + 4), (add2_x + add2_w - 6, add2_y + add2_h/2),
                     ctrl_offset=(8, 16))
    # 从前馈网络 -> Add&Norm（上层）
    add_curved_arrow(ax, (ffn_x + ffn_w/2, ffn_y + ffn_h + 2), (add2_x + 6, add2_y + add2_h/2),
                     ctrl_offset=(-12, 18))

    # 从 Add&Norm（上层） -> 主干顶部
    add_curved_arrow(ax, (add2_x + add2_w/2, add2_y + add2_h - 1), (backbone_x, outer_y + outer_h - 8),
                     ctrl_offset=(0, 20))

    # 保存图片（PNG + SVG）
    plt.savefig("transformer_block_cn.png", dpi=220, transparent=False)
    plt.savefig("transformer_block_cn.svg", dpi=220, transparent=True)
    print("✅ 已生成：transformer_block_cn.png 与 transformer_block_cn.svg")

if __name__ == "__main__":
    main()