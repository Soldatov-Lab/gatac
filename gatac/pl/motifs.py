import matplotlib.pyplot as plt
import numpy as np

def motif_enrichment(enrichment, top_motifs=10, max_cols=4, palette=None):
    n_plots = len(enrichment)
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if palette is None:
        palette = [plt.get_cmap("tab20").colors[i] for i in range(1, 20, 2)]
        palette += [plt.get_cmap("tab20").colors[i] for i in range(0, 20, 2)]

    fig, axs = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(n_cols * 5, n_rows * 5 * (top_motifs / 10)), 
        constrained_layout=True
    )

    if n_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for i, (name, vals) in enumerate(enrichment.items()):
        df = vals.sort_values("adjusted p-value").head(top_motifs)[::-1]
        axs[i].barh(
            df.name, 
            -np.log10(df["adjusted p-value"]), 
            color=palette[i % len(palette)]
        )
        axs[i].set_xlabel("-log10(adjusted p-value)")
        axs[i].set_title(f"Motif enrichment in peaks {name}")
        axs[i].grid(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    return fig, axs