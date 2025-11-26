# ======================================
# Scenario Discovery with PRIM
# ======================================

import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import seaborn as sns
from scipy.ndimage import gaussian_filter
from itertools import combinations

# --- Compatibility patches for modern versions ---
np.float = float
np.int = int
pd.DataFrame.append = lambda self, other, *args, **kwargs: pd.concat([self, other], *args, **kwargs)
pd.DataFrame.iteritems = pd.DataFrame.items

# Patch scipy.stats.binom_test -> binomtest
if not hasattr(stats, "binom_test") and hasattr(stats, "binomtest"):
    def binom_test(*args, **kwargs):
        res = stats.binomtest(*args, **kwargs)
        return res.pvalue
    stats.binom_test = binom_test


from ema_workbench.analysis import prim


# --------------------------------------
# 0 Define functions
# --------------------------------------

def load_data(input_filename, output_filename):
    
    # Load data
    X = pd.read_csv(input_filename)
    Y = pd.read_csv(output_filename)
    
    # Remove first column (index)
    X = X[X.columns[1:]]
    Y = Y[Y.columns[1:]]
    
    return X, Y

def run_prim(X, Y, target_condition, new_column_name ,threshold=0.8):
    """
    Run PRIM given X, Y, and a boolean target condition.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    Y : pd.DataFrame
        Output variables
    target_condition : pd.Series (bool)
        Boolean mask defining points of interest
    threshold : float
        Desired coverage threshold
    """

    Y[new_column_name] = target_condition
    target = target_condition == 1
    prim_alg = prim.Prim(X, target, threshold=threshold)
    box = prim_alg.find_box()
    tradeoff = box.peeling_trajectory
    
    
    return prim_alg, box, tradeoff, Y

def select_best_box(tradeoff, box, heuristic):

    # Find "elbow" manually or by heuristic
    tradeoff = tradeoff.copy()

    if heuristic == True:

        # Normalize both coverage and density to [0,1] range
        tradeoff['cov_norm'] = (tradeoff['coverage'] - tradeoff['coverage'].min()) / (tradeoff['coverage'].max() - tradeoff['coverage'].min())
        tradeoff['dens_norm'] = (tradeoff['density'] - tradeoff['density'].min()) / (tradeoff['density'].max() - tradeoff['density'].min())

        # Compute distance from (0,1) in normalized space
        # (that’s top-left corner: perfect density, full coverage)
        tradeoff['distance'] = np.sqrt((1 - tradeoff['dens_norm'])**2 + (tradeoff['cov_norm'] - 1)**2)

        # Pick the one minimizing this distance
        best_id = tradeoff['distance'].idxmin()
        best_point = tradeoff.loc[best_id]

        #print(f"📦 Best box found at id {best_id}")
        print(f"   Coverage = {best_point.coverage:.3f}")
        print(f"   Density  = {best_point.density:.3f}")

        box.select(best_id)

    else:
        best_id = tradeoff['density'].idxmax()
        best_point = tradeoff.loc[best_id]
        box.select(best_id)

    return best_id, best_point, box

def extract_active_variables(box):
    
    import re
    from io import StringIO
    import sys

    # Capture the printed output of box.inspect()
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    box.inspect()
    sys.stdout = old_stdout

    output = mystdout.getvalue()

    # Extract only the variable names in the table
    # The table lines usually have format: variable_name  min_value  max_value ...
    active_vars = re.findall(r'^\s*([a-zA-Z_]\w*)\s+[-+]?\d', output, re.MULTILINE)

    # Filter out known PRIM metrics
    prim_metrics = {'coverage', 'density', 'id', 'mass', 'mean', 'res_dim', 'box'}
    active_vars = [v for v in active_vars if v not in prim_metrics]

    return(active_vars)


def plot_pairwise_with_prim_box(X, Y, box, best_id, active_vars, output_colname,
                                name_dict=None, max_vars=10):
    """
    Plot pairwise scatterplots of PRIM active variables with box boundaries.
    """


    if len(active_vars) > max_vars:
        print(f"Too many active variables ({len(active_vars)}). Limiting to first {max_vars}.")
        active_vars = active_vars[:max_vars]

    pairs = list(combinations(active_vars, 2))
    n_plots = len(pairs)

    # Layout: one row if <=3 plots, otherwise two
    if n_plots <= 3:
        n_rows, n_cols = 1, n_plots
    else:
        n_rows, n_cols = 2, (n_plots + 1) // 2

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # --- Draw pairwise scatterplots ---
    for ax, (var1, var2) in zip(axes, pairs):
        sns.scatterplot(
            x=X[var1],
            y=X[var2],
            hue=Y[output_colname],
            palette={0: "blue", 1: "orange"},
            alpha=0.6,
            s=15,
            ax=ax,
            legend=False,  # disable individual legends
        )

        # Draw PRIM box
        v1min, v1max = box.box_lims[best_id][var1]
        v2min, v2max = box.box_lims[best_id][var2]
        rect = Rectangle(
            (v1min, v2min),
            v1max - v1min,
            v2max - v2min,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(rect)

        # Pretty labels if provided
        label1 = name_dict.get(var1, var1) if name_dict else var1
        label2 = name_dict.get(var2, var2) if name_dict else var2
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)

    # Remove unused axes
    for i in range(len(pairs), len(axes)):
        fig.delaxes(axes[i])

    # --- Unified legend (manual but consistent) ---
    legend_handles = [
        mlines.Line2D([], [], color="blue", marker="o", linestyle="None", label="Not in box (0)"),
        mlines.Line2D([], [], color="orange", marker="o", linestyle="None", label="In box (1)")
    ]

    fig.legend(
        handles=legend_handles,
        title=output_colname,
        loc="upper center",   # places it above the top row
        ncol=2,
        frameon=False
    )

    # Adjust layout to make room for the legend
    fig.tight_layout()

    if n_rows > 1:
        fig.subplots_adjust(top=0.94)  # leave some space at the top for the legend
    else:
        fig.subplots_adjust(top=0.88)

    plt.show()

    # Save including legend
    fig.savefig(f"prim_pairwise_plots_{output_colname}.png", dpi=400, bbox_inches="tight")


def plot_pairwise_heatmaps_smooth_soft_orange_blue(
    X, Y, box, best_id, active_vars, output_colname,
    name_dict=None, unit_dict = None, max_vars=10, bins=40, sigma=1.2,
    cbar_label="Share of cases = 1 (%)"
):
    """
    Smoothed heatmaps (percentage of 1s) with a soft orange → white → light-blue colormap.
    Adds a single colorbar outside the plot (0–100% scale).
    """

    plt.rcParams.update({
    "text.usetex": False,                # use built-in mathtext
    "mathtext.fontset": "dejavuserif",   # or "stix" for a TeX-like font
    "font.family": "serif",
    "axes.unicode_minus": False
    })
    
    if len(active_vars) > max_vars:
        active_vars = active_vars[:max_vars]

    pairs = list(combinations(active_vars, 2))
    n_plots = len(pairs)
    n_rows, n_cols = (1, n_plots) if n_plots <= 3 else (2, (n_plots + 1)//2)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # --- Custom soft orange → white → light blue colormap ---
    colors = ["#ffcc99", "#ffe6cc", "#ffffff", "#cce6ff", "#99ccff"]
    cmap = LinearSegmentedColormap.from_list("soft_orange_blue", colors, N=256)

    im = None  # store for colorbar reference

    for ax, (var1, var2) in zip(axes, pairs):
        x, y, z = X[var1].values, X[var2].values, Y[output_colname].values

        # Create 2D grid of values
        x_bins = np.linspace(x.min(), x.max(), bins + 1)
        y_bins = np.linspace(y.min(), y.max(), bins + 1)
        xi = np.digitize(x, x_bins) - 1
        yi = np.digitize(y, y_bins) - 1

        grid = np.full((bins, bins), np.nan)
        for i in range(bins):
            for j in range(bins):
                mask = (xi == i) & (yi == j)
                if np.any(mask):
                    grid[j, i] = np.mean(z[mask])

        # Fill NaNs and smooth
        if np.isnan(grid).all():
            grid_filled = np.zeros_like(grid)
        else:
            grid_filled = np.where(np.isnan(grid), np.nanmean(grid), grid)
        grid_smooth = gaussian_filter(grid_filled, sigma=sigma)

        # Convert to percentage
        grid_smooth *= 100

        # Plot smoothed heatmap
        im = ax.imshow(
            np.flipud(grid_smooth),
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=100
        )

        # Overlay PRIM box
        v1min, v1max = box.box_lims[best_id][var1]
        v2min, v2max = box.box_lims[best_id][var2]
        ax.add_patch(
            Rectangle((v1min, v2min), v1max - v1min, v2max - v2min,
                      edgecolor="black", facecolor="none", linewidth=2, linestyle="--")
        )

        # --- Label handling with optional units ---
        #xlabel = name_dict.get(var1, var1) if name_dict else var1
        #ylabel = name_dict.get(var2, var2) if name_dict else var2

        #if unit_dict:
        #    if var1 in unit_dict:
        ##        xlabel = f"{xlabel} [{unit_dict[var1]}]"
        #    if var2 in unit_dict:
        #        ylabel = f"{ylabel} [{unit_dict[var2]}]"

        ax.set_xlabel(name_dict.get(var1, var1), fontsize=12)
        ax.set_ylabel(name_dict.get(var2, var2), fontsize=12)




    # Remove empty axes
    for i in range(len(pairs), len(axes)):
        fig.delaxes(axes[i])

    # --- Shared colorbar outside the plots ---
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.set_ticks(np.linspace(0, 100, 6))
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()
    fig.savefig(f"prim_pairwise_heatmap_plots_paper_{output_colname}.png", dpi=400, bbox_inches="tight")




# --------------------------------------
# 1 Load data
# --------------------------------------

X, Y = load_data('sets_PAWN//inputspace.csv', 'sets_PAWN//processed_output.csv')  

# Convert some input variables to percentage scale for better interpretability
X["waterstorage_ex_loss"] = X["waterstorage_ex_loss"] * 100  # convert to percentage
X["expop_increase"] = X["expop_increase"] * 100  # convert to percentage
X["prc_area_available"] = X["prc_area_available"] * 100  # convert to percentage
X["acqueduct_losses"] = X["acqueduct_losses"] * 100  # convert to percentage


# --------------------------------------
# 2️ Define an outcome of interest and run PRIM
# --------------------------------------
baseline_cost = 13.635368 
baseline_emissions = 41.615772

target_condition = (Y['WS_new_cap'] > 0).astype(int)


#target_condition = ( (Y['monetary'] < baseline_cost) &
#                     (Y['emissions'] < baseline_emissions) ).astype(int)

#target_condition =  (Y['monetary'] < baseline_cost).astype(int) 
#target_condition =  (Y['emissions'] < baseline_emissions).astype(int) 

#target_condition = ((Y['old_des_prod'] + Y['new_des_prod']) < Y['ship_prod']).astype(int) 


#target_condition = (Y['new_des_prod'] > Y['old_des_prod']).astype(int)

# Run PRIM
prim_alg, box, tradeoff, Y = run_prim(X, Y, target_condition, 'Storage',  threshold=0.8)


# --------------------------------------
# 4 Select the best box and check results summary
# --------------------------------------
heuristic = True
best_id, best_point, box = select_best_box(tradeoff, box, heuristic)

# Visualize peeling trajectory
box.show_tradeoff()
plt.show()

# Visualize the coverage and density
box.show_ppt()  # Parallel coordinates of the found box
plt.show()

box.inspect()

# --------------------------------------
# 5️ Plot the pairwise scatterplots with PRIM box
# --------------------------------------

active_vars = extract_active_variables(box)


name_dict = {
    "expop_increase": r"Non-residential Population [%]",
    "wd_res": r"Water demand (residential) $\left[\frac{m_W^3}{\text{day}\cdot p}\right]$",
    "wd_expop": r"Water demand (expanded population) $\left[\frac{m_W^3}{\text{day}\cdot p}\right]$",
    "EF_ship": r"Ship Emission Factor $\left[\frac{kg\,CO_2}{km\cdot m_W^3}\right]$",
    "cost_water": r"Water cost $\left[\frac{€}{m_W^3}\right]$",
    "desalter_eff_old": r"Desalter efficiency (old) $\left[\frac{kWh}{m_W^3}\right]$",
    "desalter_eff_new": r"Desalter efficiency (new) $\left[\frac{kWh}{m_W^3}\right]$",
    "cost_desalter": r"Desalter cost $\left[\frac{k€}{m_W^3\cdot h}\right]$",
    "cost_fuel": r"Fuel cost $\left[\frac{€}{kWh}\right]$",
    "emission_fuel": r"Fuel emission factor $\left[\frac{kg\,CO_2}{kWh}\right]$",
    "prc_area_available": r"Rooftop Area for PV [%]",
    "pv_resource_area_per_energy_cap": r"PV area per capacity $\left[\frac{m^2}{kWp}\right]$",
    "pv_energy_cap_cost": r"PV capacity cost $\left[\frac{€}{kWp}\right]$",
    "pv_year": r"PV annual yield $\left[\frac{kWh}{kWp}\right]$",
    "acqueduct_losses": r"Aqueduct losses [\%]",
    "waterstorage_new_cap_max": r"New storage capacity [$m_W^3$]",
    "waterstorage_new_storage_cap_cost": r"New Storage cost $\left[\frac{€}{m_W^3}\right]$",
    "waterstorage_ex_loss": r"Existing storage loss [%]",
    "battery_cap_per_pv_cap": r"Battery-to-PV ratio $\left[\frac{kWh}{kWp}\right]$",
    "battery_storage_cap_cost": r"Battery storage cost $\left[\frac{€}{kWh}\right]$",
    "weight_co2": r"CO$_2$ weight [-]"
}

name_dict = {
    "expop_increase": r"$\mathit{Pop}^{NR}_{\%}\ [\%]$",
    "EF_ship": r"$\mathit{Ship}_{EF}\ \left[\frac{kg\,CO_2}{km\cdot m_W^3}\right]$",
    "cost_fuel": r"$\mathit{Fuel}_{cost}\ \left[\frac{€}{kWh}\right]$",
    "emission_fuel": r"$\mathit{Fuel}_{EF}\ \left[\frac{kg\,CO_2}{kWh}\right]$",
    "prc_area_available": r"$\mathit{PV}_{area}\ [\%]$",
    "waterstorage_ex_loss": r"$\mathit{WS}_{loss}^{ex}\ [\%]$",
    "weight_co2": r"$\mathit{W}_{CO_2}\ [-]$"
}





plot_pairwise_heatmaps_smooth_soft_orange_blue(
    X, Y, box, best_id, active_vars, 'Storage' , name_dict=name_dict, cbar_label="Solutions with new storage installation [%]"
)

