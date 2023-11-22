import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

# matplotlib.use("Qt5Agg")

# Set font sizes: set according to LaTEX document
SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

# LaTex document properties
textwidth = 448
textheight = 635.5
# plt.rc("text", usetex='True')
plt.rc('font', family='serif', serif='Arial')

# Set plot properties:
# plt.rc('figure', figsize=(15, 15))
plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title
plt.rc('axes', axisbelow=True)


def plot_size(fraction=1, height_fraction=0):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
        Parameters
        ----------
        :param fraction: float, optional (Fraction of the width which you wish the figure to occupy)
        :param height_fraction: float, optional (Fraction of the entire page height you wish the figure to occupy)

        Returns
        -------
        fig_dim: tuple (Dimensions of figure in inches)
        """
    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if height_fraction == 0 :
        # Figure height in inches: golden ratio
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = textheight * inches_per_pt * height_fraction

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_1d_gpe_final(gpr_x, gpr_y, conf_int_lower, conf_int_upper, tp_x, tp_y, tp_init, it, true_y, true_x=None,
                      analytical=None):
    """
        Plots the final GPE, including GPE predictions and all BAL-selected training points and confidence intervals.

        Parameters
        ----------
        :param gpr_x: np.array [mc_size, 1], parameter values where GPE was evaluated (X)
        :param gpr_y: np.array [N.Obs, mc_size], GPE prediction values (Y)
        :param conf_int_lower: np.array [N.obs, mc_size], 95% lower confidence intervals for each gpr_y from GPE
        :param conf_int_upper: np.array [N.obs, mc_size], 95% upper confidence intervals for each gpr_y from GPE
        :param tp_x: np.array [#tp+1, 1], parameter values from training points
        :param tp_y: np.array [#tp+1, N.Obs], forward model outputs at tp_x
        :param tp_init: int, number of initial training points
        :param true_y: np.array [1, N.Obs] Y value of the observation value
        :param true_x: np.array [1, 1] true parameter value or None (if not available)
        :param analytical: np.array [2, mc_size], x and y values evaluated in forward model or None (to not plot it)
        :param it: int , number of iteration, for plot title
        """
    n_o = gpr_y.shape[0]
    del_last = False
    if n_o == 1:
        rows = 1
        cols = 1
    elif (n_o % 2) == 0:
        rows = int(n_o/2)
        cols = 2
    elif (n_o % 3) == 0:
        rows = int(n_o / 3)
        cols = 3
    else:
        rows = math.ceil(n_o / 2)
        cols = 2
        del_last = True

    fig, ax = plt.subplots(rows, cols, sharex=True)
    if del_last:
        ax[rows-1, cols-1].remove()

    if n_o > 1:
        loop_item = ax.reshape(-1)
    else:
        loop_item = np.array([ax])

    for o, ax_i in enumerate(loop_item):
        # Get overall limits:
        # GPR limits:
        lims_gpr = np.array([math.floor(np.min(conf_int_lower)), math.ceil(np.max(conf_int_upper))])
        # Get limits
        lims_x = [math.floor(np.min(gpr_x)), math.ceil(np.max(gpr_x))]

        if analytical is not None:
            # order analytical evaluations
            a_data = analytical[:, analytical[0, :].argsort()]
            # Analytical limits
            lims_an = np.array([math.floor(np.min(analytical[1, :])), math.ceil(np.max(analytical[1, :]))])
            # Max limit is defined by gpr and analytical
            lims_y = [np.min(np.array([lims_gpr[0], lims_an[0]])), np.max(np.array([lims_gpr[1], lims_an[1]]))]
        else:
            lims_y = lims_gpr

        # GPE data ---------------------------------------------------------------
        data = np.vstack((gpr_x.T, gpr_y[o, :].T, conf_int_lower[o, :], conf_int_upper[o, :]))
        gpr_data = data[:, data[0, :].argsort()]

        ax_i.plot(gpr_data[0, :], gpr_data[1, :], label="GPE mean", linewidth=1, color='b', zorder=1)
        ax_i.fill_between(gpr_data[0, :].ravel(), gpr_data[2, :], gpr_data[3, :], alpha=0.5,
                          label=r"95% confidence interval")

        # TP ---------------------------------------------------------------------
        # Color map:
        cm = plt.cm.get_cmap('RdYlBu')
        cls = []
        tot = tp_x.shape[0]-tp_init+1
        for i in range(1, tp_x.shape[0]-tp_init+1):
            cls.append(cm(i/tot))

        ax_i.scatter(tp_x[0:tp_init], tp_y[0:tp_init, o], label="TP", color="k", zorder=2)  # s=500
        ax_i.scatter(tp_x[tp_init:], tp_y[tp_init:, o], color=cls, marker="x", zorder=2, linewidths=3, label="BAL TP")  # s=800

        # Plot observation ----------------------------------------------------------------
        if true_x is not None:
            ax_i.scatter(true_x[0, 0], true_y[0, o], marker="*", color="g", zorder=2)  # s=500
        else:
            ax_i.plot([np.min(gpr_x), np.max(gpr_y)], [true_y[0, o], true_y[0, o]], color="g", linestyle="dotted")

        # Plot analytical data
        if analytical is not None:
            ax.plot(a_data[0, :], a_data[1, :], color='k', linewidth=1, linestyle="dashed", label="f(x)", zorder=1)

        # Legends
        handles, labels = ax_i.get_legend_handles_labels()

        # Set limits --------------------------------------------------------------------------------
        ax_i.set_xlim(lims_x)
        ax_i.set_ylim([lims_y[0] - (0.5 * (lims_y[1] - lims_y[0])), lims_y[1] * 1.1])

        # set labels ------------------------------------------------------------------------------------
        # ax_i.set_xlabel("Parameter value")
        # ax_i.set_ylabel("Y")
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)

    if n_o > 4:
        fig.text(0.5, 0.1, "Parameter value", ha='center')
        fig.text(0.04, 0.5, "Model output", ha='center', rotation='vertical')
    elif n_o == 1:
        ax_i.set_xlabel("Parameter value")
        ax_i.set_ylabel("Y")

    fig.suptitle(f'Iteration={it + 1}')
    fig.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.3, hspace=0.3)
    fig.legend(handles, labels, loc='lower center', ncol=5)
    plt.show(block=False)


def plot_gpe_scores(bme_values, re_values, tp_i, ref_bme=None, ref_re=None, title=None):
    """
    Function plots BME, (ELPD) and RE for the GPE and, if available, the reference values from the full-complexity
    forward model

    Parameter:
    :param bme_values: np.array [number of GPE training iterations, 1], with BME values for each GPE generated during
    training process
    :param re_values: np.array [number of GPE training iterations, 1], with RE values for each GPE generated during
    training process
    :param tp_i: int, number of initial training points
    :param ref_bme: int, BME of reference model (if available)
    :param ref_re: int, RE of reference model (if available)
    :param title: str, with title of the figure (if user decides is necessary)
    :return: None
    """

    n_iter = bme_values.shape[0]
    x = np.arange(tp_i, n_iter + tp_i)
    # re_plot = re_values[0:n_iter, 0]
    # bme_plot = bme_values[0:n_iter, 0]

    fig, ax = plt.subplots(2, 1, figsize=plot_size(fraction=1), sharex=True)  # , height_fraction=0.5))

    ax[0].plot(x, bme_values[0:n_iter, 0], label="BME (GPE)")
    if ref_bme is not None:
        ax[0].plot(x, np.full(x.shape[0], ref_bme), label="ref BME")

    ax[0].set_title("BME vs GPR training iterations")
    # ax[0].set_xlabel("$N_{TP}$")
    ax[0].set_ylabel("BME")
    ax[0].set_yscale('log')

    ax[1].plot(x, re_values[0:n_iter, 0], label="RE (GPE)")
    if ref_re is not None:
        ax[1].plot(x, np.full(x.shape[0], ref_re), label="ref RE")

    ax[1].set_title("RE vs GPR training iterations")
    ax[1].set_xlabel("$N_{TP}$")
    ax[1].set_ylabel("RE")

    if title is not None:
        fig.suptitle(title)

    plt.subplots_adjust(top=0.86, bottom=0.13, wspace=0.1, hspace=0.4)
    plt.show(block=False)


def plot_bal_criteria(bal_criteria, score_name):
    """
    Function plots BME and RE for the GPE and, if available, the reference values from the full-complexity
    forward model
    :param bal_criteria: np.array [number of BAL training iterations, 1], with BME values for each GPE generated during
    training process
    :param score_name: str, name of the score used as BAl criteria
    """

    n_iter = bal_criteria.shape[0]
    x = np.arange(1, n_iter + 1)

    fig, ax = plt.subplots(figsize=plot_size(fraction=1))  # , height_fraction=0.5))

    ax.plot(x, bal_criteria, label="BME (GPE)", marker='o')
    ax.set_title(f"BAL criteria vs No. BAL training iterations")
    ax.set_xlabel("$N_{TP}$")
    ax.set_ylabel(score_name)

    plt.subplots_adjust(top=0.86, bottom=0.13, wspace=0.1, hspace=0.4)
    plt.show(block=False)
