#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import numpy
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from utils import compute_fmr_thresholds, split_scores_by_variable, farfrr, load_scores, compute_werm_geo

def compute_fmr_fnmr_tradeoff(
    negatives,
    positives,
    variable_suffix,
    fmr_thresholds,
    fair_fn=compute_werm_geo,
    pre_computed_taus=None,
):
    """
    Compute the fmr and fnmr per demographic and fairness

    Parameters
    ----------
      negatives: dataframe
        Pandas Dataframe containing the negative scores (or impostor scores, or even non-mated scores)

      positives: dataframe
        Pandas Dataframe containing the positive scores (or genuines scores, or even mated scores)

      variable_suffix: str
        The suffix of a variable that will be appended to `bio_ref_[variable_suffix]` for biometric references
        and `probe_[variable_suffix]` that will be appended to probes.

      fmr_thresholds: list
        List containing the FMR operational points
        
      fair_fn: function
        Function computing the fairness and its components

      pre_compute_taus: list
        Set of precomputed decision thresholds. If `None`, it will be computed on the fly

    """

    if pre_computed_taus is None:
        taus = (
            compute_fmr_thresholds(negatives, fmr_thresholds)
            if pre_computed_taus
            else pre_computed_taus
        )
    else:
        taus = pre_computed_taus


    negatives_as_dict, positives_as_dict = split_scores_by_variable(
        negatives, positives, variable_suffix
    )
    # Iterating ONLY on comparisons of the same
    # demographic group
    fmrs = dict()
    fnmrs = dict()
    fmrs_scaled = dict()
    fnmrs_scaled = dict()

    # for key in positives_as_dict:
    for key in negatives_as_dict:
     
        fmrs[key] = []
        if key in positives_as_dict:
            fnmrs[key] = []
        for t in taus:

            if key in positives_as_dict:
                fmr, fnmr = farfrr(
                    negatives_as_dict[key]["score"].compute().to_numpy(),
                    positives_as_dict[key]["score"].compute().to_numpy(),
                    t,
                )
                fnmrs[key].append(fnmr)
            else:
                if negatives_as_dict[key]["score"].compute().to_numpy().size == 0:
                    fmr = float('NaN')
                else:
                    fmr, _ = farfrr(
                        negatives_as_dict[key]["score"].compute().to_numpy(),
                        [0.0],
                        t,
                    )
            fmrs[key].append(fmr)

    # Stacking vertically FMRS and FNMRs so we can compute FDR
    A_tau = numpy.vstack([fnmrs[k] for k in fnmrs]).T
    B_tau = numpy.vstack([fmrs[k] for k in fnmrs]).T

    fmrs_scaled = []
    fnmrs_scaled = []
    fairness = []
    for a_tau, b_tau in zip(A_tau, B_tau):
        fnmr_scaled, fmr_scaled, fair = fair_fn(a_tau, b_tau)
        fmrs_scaled.append(fmr_scaled)
        fnmrs_scaled.append(fnmr_scaled)
        fairness.append(fair)

    return fmrs, fnmrs, fmrs_scaled, fnmrs_scaled, fairness

def plot_demographic_boxplot(
    negatives,
    positives,
    variable_suffix,
    fmr_thresholds=None,
    label_lookup_table=None,
    title="",
    pre_computed_taus=None,
):
    """
    Plot the box-plots of the score distribution

    Parameters
    ----------
      negatives: dataframe
        Pandas Dataframe containing the negative scores (or impostor scores, or even non-mated scores)

      positives: dataframe
        Pandas Dataframe containing the positive scores (or genuines scores, or even mated scores)

      variable_suffix: str
        The suffix of a variable that will be appended to `bio_ref_[variable_suffix]` for biometric references
        and `probe_[variable_suffix]` that will be appended to probes.

      fmr_thresholds: list
        List containing the FMR operational points

      label_lookup_table: dict
         Lookup table mapping `variable` to the actual label of the variable

      title: str
        Plot title

      pre_compute_taus: list
        Set of precomputed decision thresholds. If `None`, it will be computed on the fly

    """

    # Computing decision thresholds if we have any FMR
    if pre_computed_taus is None:
        taus = (
            compute_fmr_thresholds(negatives, fmr_thresholds)
            if fmr_thresholds is not None
            else None
        )
    else:
        taus = pre_computed_taus

    # Spliting the scores by cohort
    negatives_as_dict, positives_as_dict = split_scores_by_variable(
        negatives, positives, variable_suffix
    )


    def _color_boxplot(bp, color):
        for patch in bp["boxes"]:
            patch.set_facecolor(color)

    def _get_scores(negatives_as_dict, positives_as_dict):
        """
        Getting the scores as numpy arrays,
        so we can plot using matplotlib
        """
       
        scores = dict()
        for n in negatives_as_dict:

            negatives = negatives_as_dict[n]["score"].compute().to_numpy()

            positives = (
                positives_as_dict[n]["score"].compute().to_numpy()
                if n in positives_as_dict
                else []
            )

            scores[n] = [negatives, positives]        
        return scores

    def _plot(scores, axes, labels):
        # This code raises a warning
        # https://github.com/matplotlib/matplotlib/issues/16353
        bp_negatives = axes.boxplot(
            [scores[s][0] for s in scores],
            patch_artist=True,
            labels=labels,
            showfliers=False,
            vert=False,
        )

        _color_boxplot(bp_negatives, "tab:red")

        bp_positives = axes.boxplot(
            [scores[s][1] for s in scores],
            patch_artist=True,
            labels=labels,
            showfliers=False,
            vert=False,
        )

        _color_boxplot(bp_positives, "tab:blue")

    # Matching the variable values to
    # the actual labels for readability
    # labels = list(positives_as_dict.keys())
    labels = list(negatives_as_dict.keys())
    if label_lookup_table is not None:
        labels = [label_lookup_table[l] for l in labels]

    # Plotting the boxplots
    fig, ax = plt.subplots(figsize=(16, 8))

    fig.suptitle(title)

    axes = plt.subplot(1, 1, 1)

    def _compute_scores_and_plot(
        negatives_as_dict, positives_as_dict, axes, plot_fmrs=True
    ):

        # Computing the scores
        scores = _get_scores(negatives_as_dict, positives_as_dict)
        # Plotting the box plot
        _plot(scores, axes, labels)
        plt.grid(True)
        plt.yticks(fontsize=18)
        if plot_fmrs:
            if taus is not None:
                colors = list(plt.cm.get_cmap("tab20").colors)

                [
                    axes.axvline(
                        t, linestyle="--", label="$\\tau=FMR_{" + str(f) + "}$", color=c
                    )
                    for t, c, f in zip(taus, colors, fmr_thresholds)
                ]

    _compute_scores_and_plot(negatives_as_dict, positives_as_dict, axes)

    fig.legend(loc=2, fontsize=16)

    return fig


def plot_fmr_fnmr_ratio(
    negatives,
    positives,
    variable_suffix,
    fmr_thresholds,
    label_lookup_table=None,
    title="False Match and False non Match trade-off per demographic",
    fair_fn=compute_werm_geo,
    pre_computed_taus=None,
):

    """
    Compute fairness (including fmr and fnmr tradeoff) and make a tradeoff plot

    Parameters
    ----------
      negatives: dataframe
        Pandas Dataframe containing the negative scores (or impostor scores, or even non-mated scores)

      positives: dataframe
        Pandas Dataframe containing the positive scores (or genuines scores, or even mated scores)

      variable_suffix: str
        The suffix of a variable that will be appended to `bio_ref_[variable_suffix]` for biometric references
        and `probe_[variable_suffix]` that will be appended to probes.

      fmr_thresholds: list
        List containing the FMR operational points

      label_lookup_table: dict
         Lookup table mapping `variable` to the actual label of the variable

      title: str
        Plot title

      fair_fn: function
        Function computing the fairness and its components

      pre_compute_taus: list
        Set of precomputed decision thresholds. If `None`, it will be computed on the fly

    """

    fmrs, fnmrs, fmrs_scaled, fnmrs_scaled, fairness = compute_fmr_fnmr_tradeoff(
        negatives,
        positives,
        variable_suffix,
        fmr_thresholds,
        fair_fn,
        pre_computed_taus,
    )

    fig, ax = plt.subplots(figsize=(16, 8))


    title = f"System: {title}. FMR and FNMR trade-off"

    fig.suptitle(title)

    # LABELS FOR FNMR
    labels_fnmr = list(fnmrs.keys())
    if label_lookup_table is not None:
        labels_fnmr = [label_lookup_table[l] for l in labels_fnmr]

    # Plot FNMR
    axes = plt.subplot(2, 1, 1)
    [
        plt.semilogx(fmr_thresholds, fnmrs[f], label=l)
        for f, l in zip(fnmrs, labels_fnmr)
    ]
    plt.ylabel("$FNMR(\\tau)$", fontsize=18)

    plt.grid(True)

    # LABELS FOR FMR
    labels_fmr = list(fmrs.keys())
    if label_lookup_table is not None:
        labels_fmr = [label_lookup_table[l] for l in labels_fmr]

    axes = plt.subplot(2, 1, 2)
    [plt.semilogx(fmr_thresholds, fmrs[f], label=l) for f, l in zip(fmrs, labels_fmr)]
    plt.ylabel("$FMR(\\tau)$", fontsize=18)
    plt.xlabel("$\\tau=FMR_{10^{-x}}$", fontsize=18)

    plt.grid(True)
    plt.legend()

    return fig, fmrs, fnmrs, fmrs_scaled, fnmrs_scaled, fairness

def plot_scaled_fnmrs_fmrs_tradeoff(
    fmr_thresholds,
    title=None,
    cached_fnmrs=None,
    cached_fmrs=None,
):
    """
    Plot scaled tradeoff

    Parameters
    ----------
      fmr_thresholds: list
        List containing the FMR operational points

      title: str
        Plot title
        
      cached_fnmrs: list
        Set of precomputed fnmrs. 

      cached_fmrs: list
        Set of precomputed fmrs. 
    """
    title = f"System: {title}. (Scaled) FMR and FNMR trade-off"
    
    rates = {"fnmr": cached_fnmrs, "fmr":cached_fmrs}
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(title)
    axes = plt.subplot(2, 1, 1)

    [plt.semilogx(fmr_thresholds, rates["fnmr"], label="fnmr")]
    plt.ylabel("$FNMR(Scaled)(\\tau)$", fontsize=18)

    plt.grid(True)
    
    axes = plt.subplot(2, 1, 2)
    [plt.semilogx(fmr_thresholds, rates["fmr"], label="fmr")]
    plt.ylabel("$FMR(Scaled)(\\tau)$", fontsize=18)
    plt.xlabel("$\\tau=FMR_{10^{-x}}$", fontsize=18)

    plt.grid(True)

    plt.xticks(fontsize=14,)
    plt.yticks(fontsize=14,)

    return fig


def plot_fair(
    fmr_thresholds,
    labels,
    cached_fairness=None,
):
    """
    Plot fairness 

    Parameters
    ----------
      fmr_thresholds: list
        List containing the FMR operational points
        
      labels: list
        List containing the labels for each line 

      cached_fairness: list
        Set of precomputed fairness values. 
    """

    fig, ax = plt.subplots(figsize=(8,6))

    [
        plt.semilogx(fmr_thresholds, f, label=l, linewidth=2)
        for f, l in zip(cached_fairness, labels)
    ]
    [plt.scatter(fmr_thresholds, f) for f in cached_fairness]

    plt.ylabel("$WERM(\\tau)$", fontsize=18)
    plt.xlabel("$\\tau=FMR_{10^{-x}}$", fontsize=18)
    plt.xticks(fontsize=14,)
    plt.yticks(fontsize=14,)
    plt.grid(True)
    fig.suptitle("Fairness Plot")

    plt.legend(prop={'size': 18})
   # plt.tight_layout()
    return fig


def standard_report(
    negatives,
    positives,
    output_filename,
    variable_suffix,
    fmr_thresholds=[10 ** i for i in list(range(-6, 0))],
    titles=None,
    lookup_table=None,
    fair_fn=compute_werm_geo,
):
    """
    Generate standard report for given scores, including, for each score file, a box plot, a tradeoff plot of fnmr and fmr, a tradeoff plot for scaled fnmr and fmr, and finally, a summary fairness plot for all scores.

    Parameters
    ----------
      negatives: List of dataframes
        List of Pandas Dataframes containing the negative scores (or impostor scores, or even non-mated scores)

      positives: List of dataframes
        List of Pandas Dataframes Pandas Dataframe containing the positive scores (or genuines scores, or even mated scores)

      output_filename: str
        Output name of the report
    
      variable_suffix: str
        The suffix of a variable that will be appended to `bio_ref_[variable_suffix]` for biometric references
        and `probe_[variable_suffix]` that will be appended to probes.

      fmr_thresholds: list
        List containing the FMR operational points
        
      titles: list
        List containing titles for each score file
      
      lookup_table: dict
        Dictionary containing demographic pairs

      fair_fn: function
        Function computing the fairness and its components

    """

    pdf = PdfPages(output_filename)

    file_name, _ = output_filename.split(".pdf")
    # txt_name = name+".txt"
    fairness_dict = defaultdict(dict)
    cached_fairness = []
    if titles is not None:
        assert len(titles) == len(negatives)

    taus = []
    for i, (n, p) in enumerate(
        zip(negatives, positives)
    ):
 
        n = n.persist()
        p = p.persist()

        # Computing the decision thresholds
        tau = compute_fmr_thresholds(n, fmr_thresholds)
        taus.append(tau)

        title = None if titles is None else titles[i]


        fig = plot_demographic_boxplot(
            negatives=n,
            positives=p,
            variable_suffix=variable_suffix,
            label_lookup_table=lookup_table,
            fmr_thresholds=fmr_thresholds,
            title=title,
            pre_computed_taus=tau,
        )

        pdf.savefig(fig)

        #### PLOTTING THE FMR AND FNMR TRADE OFF
        fig, fmrs, fnmrs, fmrs_scaled, fnmrs_scaled, fairness = plot_fmr_fnmr_ratio(
            n,
            p,
            variable_suffix=variable_suffix,
            fmr_thresholds=fmr_thresholds,
            label_lookup_table=lookup_table,
            title=title,
            pre_computed_taus=tau,
            fair_fn=fair_fn,
        )
        # breakpoint()
        fairness_dict[title]["fairness"] = fairness
        fairness_dict[title]["fmrs"] = fmrs
        fairness_dict[title]["fnmrs"] = fnmrs
        fairness_dict[title]["fmrs_scaled"] = fmrs_scaled
        fairness_dict[title]["fnmrs_scaled"] = fnmrs_scaled

        cached_fairness.append(fairness)

        pdf.savefig(fig)
    
        fig = plot_scaled_fnmrs_fmrs_tradeoff(
            fmr_thresholds,
            title,
            cached_fnmrs=fnmrs_scaled,
            cached_fmrs=fmrs_scaled
        )
        pdf.savefig(fig)
    # Plotting the Fairness
    fig = plot_fair(
        fmr_thresholds,
        titles,
        cached_fairness=cached_fairness,
    )
    pdf.savefig(fig)
    
    pdf.close()
    
    import json
    with open(file_name+".json","a+") as f:
        json.dump(fairness_dict, f)

def rfw_report(
    scores,
    output_filename,
    fmr_thresholds=[10 ** i for i in list(range(-3, 0))],
    titles=None,
    fair_fn=compute_werm_geo,
):
    scores = scores.split(",")
    titles = titles.split(",")
    variables = {
        "Asian": "Asian",
        "African": "African",
        "Caucasian": "Caucasian",
        "Indian": "Indian",
    }

    lookup_table = dict()
    for a in list(variables.keys()):
        for b in list(variables.keys()):
            lookup_table[f"{a}__{b}"] = f"{variables[a]}-{variables[b]}"
 
    variable_suffix = "race"

    negatives, positives = load_scores(scores)

    standard_report(
        negatives,
        positives,
        output_filename,
        variable_suffix,
        fmr_thresholds=fmr_thresholds,
        titles=titles,
        lookup_table=lookup_table,
        fair_fn=fair_fn,
    )

def vgg2_report(
    scores,
    output_filename,
    variable_suffix,
    fmr_thresholds=[10 ** i for i in list(range(-4, 0))],
    fair_fn=compute_werm_geo,
    titles=None,
):
    scores = scores.split(",")
    titles = titles.split(",")
    if variable_suffix == "race":
        variables = {
            "A": "Asian",
            "B": "Black",
            "I": "Indian",
            "W": "White",
        }
    elif variable_suffix == "gender":
        variables = {
            "m": "Male",
            "f": "Female",
        }

    lookup_table = dict()
    for a in list(variables.keys()):
        for b in list(variables.keys()):
            lookup_table[f"{a}__{b}"] = f"{variables[a]}-{variables[b]}"


    negatives, positives = load_scores(scores)

    standard_report(
        negatives,
        positives,
        output_filename,
        variable_suffix,
        fmr_thresholds=fmr_thresholds,
        titles=titles,
        lookup_table=lookup_table,
        fair_fn=fair_fn,
    )




import argparse
def get_args(command_line_options = None):
    
    parser = argparse.ArgumentParser("WERM_report",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset","-d", default = "rfw", type=str, help = "Dataset, possible choices: rfw, vgg2") 
    parser.add_argument("--scores_directory","-dir", default = None, type=str, help = "A comma-separated list of directories for csv scores files")
    parser.add_argument("--variable_suffix","-var", default = "race", type=str, help = "Demographic group name used to compute fairness, used only for VGGFace2 dataset, posible choices: race, gender")
    parser.add_argument("--titles","-t", default = None, type=str, help = "A comma-separated list of titles correspond to the scores")
    parser.add_argument("--output_filename","-o", default = None, type=str, help = "OUTPUT pdf file path, to save all generated plots")

    args = parser.parse_args(command_line_options)

    return args


if __name__ == '__main__':
    args = get_args()
    if args.dataset == "rfw":
        rfw_report(
            scores = args.scores_directory,
            output_filename = args.output_filename,
            titles=args.titles,
        )
    elif args.dataset == "vgg2":
        vgg2_report(
            scores = args.scores_directory,
            output_filename = args.output_filename,
            variable_suffix = agrs.variable_suffix,
            titles=args.titles,
        )
