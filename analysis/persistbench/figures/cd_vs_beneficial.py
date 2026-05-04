
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 9.5,
    'axes.labelsize': 14.5,
    'xtick.labelsize': 14.5,
    'ytick.labelsize': 14.5,
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

OUTDIR = Path('cd_vs_beneficial_scatter_boxed_bold')
OUTDIR.mkdir(exist_ok=True)

# Fixed method order
method_keys = ['Flat Memory List', 'Fixed Partitions', 'Dynamic Partitions', '2-Level Tree']
method_labels = ['List', 'Fixed partitions', 'Dynamic partitions', 'Tree']

# Polished color set
method_colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']

# Standard prompt values: x = beneficial-memory failure, y = cross-domain failure
DATA = {
    'Llama 3.3-70B': {
        'Flat Memory List': {'CD': 13.0, 'Beneficial': 59.0},
        'Fixed Partitions': {'CD': 9.0, 'Beneficial': 61.0},
        'Dynamic Partitions': {'CD': 10.5, 'Beneficial': 56.0},
        '2-Level Tree': {'CD': 9.0, 'Beneficial': 64.0},
    },
    'Qwen 3-235B': {
        'Flat Memory List': {'CD': 80.0, 'Beneficial': 22.0},
        'Fixed Partitions': {'CD': 71.0, 'Beneficial': 17.0},
        'Dynamic Partitions': {'CD': 66.5, 'Beneficial': 22.0},
        '2-Level Tree': {'CD': 70.5, 'Beneficial': 14.0},
    },
    'DeepSeek V3.2': {
        'Flat Memory List': {'CD': 69.0, 'Beneficial': 21.0},
        'Fixed Partitions': {'CD': 60.5, 'Beneficial': 35.0},
        'Dynamic Partitions': {'CD': 55.0, 'Beneficial': 16.0},
        '2-Level Tree': {'CD': 67.5, 'Beneficial': 20.0},
    },
    'GPT-OSS 120B': {
        'Flat Memory List': {'CD': 51.0, 'Beneficial': 21.0},
        'Fixed Partitions': {'CD': 44.5, 'Beneficial': 19.0},
        'Dynamic Partitions': {'CD': 41.0, 'Beneficial': 16.0},
        '2-Level Tree': {'CD': 44.5, 'Beneficial': 16.0},
    },
    'GLM-4.7': {
        'Flat Memory List': {'CD': 58.5, 'Beneficial': 19.0},
        'Fixed Partitions': {'CD': 47.5, 'Beneficial': 15.0},
        'Dynamic Partitions': {'CD': 48.0, 'Beneficial': 14.0},
        '2-Level Tree': {'CD': 55.0, 'Beneficial': 17.0},
    },
    'Grok 4.1 Fast': {
        'Flat Memory List': {'CD': 57.5, 'Beneficial': 21.0},
        'Fixed Partitions': {'CD': 49.0, 'Beneficial': 24.0},
        'Dynamic Partitions': {'CD': 51.5, 'Beneficial': 19.0},
        '2-Level Tree': {'CD': 51.5, 'Beneficial': 15.0},
    },
    'Gemini 3.1 Pro': {
        'Flat Memory List': {'CD': 64.5, 'Beneficial': 0.0},
        'Fixed Partitions': {'CD': 70.5, 'Beneficial': 0.0},
        'Dynamic Partitions': {'CD': 59.5, 'Beneficial': 2.0},
        '2-Level Tree': {'CD': 67.0, 'Beneficial': 1.0},
    },
}

def filename_safe(name: str) -> str:
    return name.lower().replace(' ', '_').replace('.', '').replace('-', '_')

def is_relevant_point(label, x, y, baseline_x, baseline_y):
    """
    Defines which data labels should be visually emphasized.

    Current rule:
    - Bold the flat-list baseline and transformed methods that reduce cross-domain leakage
      relative to the flat list.
    - This highlights the main paper claim: structured memory reduces CD failure.

    Stricter Pareto rule, if preferred:
    return label != 'List' and (y < baseline_y) and (x <= baseline_x)
    """
    return label == 'List' or y < baseline_y

def style_axis(ax):
    # Fully boxed axes
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color('#555555')

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=3.2,
        width=0.8,
        colors='#444444'
    )

    ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.20)

    # Better axis-label styling
    ax.xaxis.label.set_fontweight('semibold')
    ax.yaxis.label.set_fontweight('semibold')

def draw_points(ax, xs, ys, point_size, label_fontsize, label_offset, label_sides=None):
    baseline_x, baseline_y = xs[0], ys[0]
    if label_sides is None:
        label_sides = ['right'] * len(method_labels)

    for x, y, c, label, side in zip(xs, ys, method_colors, method_labels, label_sides):
        relevant = is_relevant_point(label, x, y, baseline_x, baseline_y)

        # Relevant methods get larger markers and stronger outlines.
        ax.scatter(
            [x], [y],
            s=point_size + (18 if relevant else 0),
            color=c,
            edgecolor='#222222' if relevant else 'white',
            linewidth=0.9 if relevant else 0.55,
            zorder=3
        )

        x_side = side.split('_')[-1] if '_' in side else side
        y_side = side.split('_')[0] if '_' in side else 'top'

        x_off = label_offset if x_side == 'right' else -label_offset
        y_off = -label_offset if y_side == 'bottom' else label_offset
        ha = 'left' if x_side == 'right' else 'right'
        va = 'top' if y_side == 'bottom' else 'bottom'

        ax.text(
            x + x_off,
            y + y_off,
            label,
            fontsize=label_fontsize,
            fontweight='bold' if relevant else 'normal',
            ha=ha,
            va=va,
            color='#111111' if relevant else '#333333',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='white',
                edgecolor='#222222' if relevant else 'none',
                linewidth=0.45 if relevant else 0,
                alpha=0.84 if relevant else 0.72
            )
        )

# Individual figures
for model, vals in DATA.items():
    fig, ax = plt.subplots(figsize=(3.7, 3.7))

    xs = [vals[k]['Beneficial'] for k in method_keys]
    ys = [vals[k]['CD'] for k in method_keys]

    xmin = min(xs) - 3
    xmax = max(xs) + 3
    ymin = min(ys) - 3
    ymax = max(ys) + 3

    draw_points(ax, xs, ys, point_size=48, label_fontsize=7.8, label_offset=0.30)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Beneficial-memory failure (%)', fontsize=10, labelpad=5)
    ax.set_ylabel('Cross-domain leakage (%)', fontsize=10, labelpad=5)
    ax.set_title(model, fontsize=9.2, fontweight='semibold', pad=5, color='#222222')

    style_axis(ax)

    fig.tight_layout()
    stem = f'{filename_safe(model)}_cd_vs_beneficial_scatter_boxed_bold'
    # fig.savefig(OUTDIR / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / f'{stem}.png', bbox_inches='tight')
    plt.close(fig)

# Combined preview grid: keep only selected models, as in your current version.
fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.8))
axes = axes.flatten()

models_subset = ["DeepSeek V3.2", "GPT-OSS 120B", "GLM-4.7", "Qwen 3-235B"]

for ax, model in zip(axes, models_subset):
    vals = DATA[model]
    xs = [vals[k]['Beneficial'] for k in method_keys]
    ys = [vals[k]['CD'] for k in method_keys]

    xmin = min(xs) - 3
    xmax = max(xs) + 3
    ymin = min(ys) - 3
    ymax = max(ys) + 3

    if model == "DeepSeek V3.2":
        draw_points(ax, xs, ys, point_size=25, label_fontsize=12, label_offset=0.3,
            label_sides=['right', 'left', 'right', 'bottom_right'])
    elif model == "GLM-4.7":
        draw_points(ax, xs, ys, point_size=25, label_fontsize=12, label_offset=0.3,
                label_sides=['right', 'bottom_right', 'right', 'left'])
    elif model == "Qwen 3-235B":
        draw_points(ax, xs, ys, point_size=25, label_fontsize=12, label_offset=0.3,
                label_sides=['right', 'right', 'left', 'left'])
    else:
        draw_points(ax, xs, ys, point_size=25, label_fontsize=12, label_offset=0.3,
                label_sides=['right', 'right', 'right', 'right'])
    

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Beneficial-memory failure (%)', fontsize=17.8, labelpad=4)
    ax.set_ylabel('Cross-domain leakage (%)', fontsize=17.8, labelpad=4)
    ax.set_title(model, fontsize=11.5, fontweight='semibold', pad=4, color='#222222')

    style_axis(ax)

fig.tight_layout()
# fig.savefig(OUTDIR / 'combined_preview_grid.pdf', bbox_inches='tight')
fig.savefig(OUTDIR / 'combined_preview_grid.png', bbox_inches='tight')
plt.close(fig)

print(f'Figures saved to: {OUTDIR.resolve()}')
