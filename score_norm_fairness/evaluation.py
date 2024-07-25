import numpy

def load(filename):
    import dask.dataframe

    df = dask.dataframe.read_csv(filename, 100 * 1024 * 1024)
    
    genuines = df[df.probe_subject_id == df.bio_ref_subject_id]
    impostors = df[df.probe_subject_id != df.bio_ref_subject_id]

    return impostors, genuines


def load_scores(scores):

    negatives = []
    positives = []

    for i, d in enumerate(scores):

        n, p = load(d)
        negatives.append(n)
        positives.append(p)

    return negatives, positives


def split_scores_by_variable(negatives, positives, variable_suffix, samevariable=False):


    bio_ref_variable = f"bio_ref_{variable_suffix}"
    probe_variable = f"probe_{variable_suffix}"

    # Getting all possible values (using the negatives as a reference)
    bio_ref_cohorts = negatives[bio_ref_variable].unique()
    probe_cohorts = negatives[probe_variable].unique()

    negatives_as_dict = dict()
    positives_as_dict = dict()

    def filter(df, variable_suffix, probe_value, bio_ref_value):
        return df.loc[
            (df[f"probe_{variable_suffix}"] == probe_value)
            & (df[f"bio_ref_{variable_suffix}"] == bio_ref_value)
        ]

    for b in bio_ref_cohorts:
        for p in probe_cohorts:
            if b == p:
                positives_filtered = filter(positives, variable_suffix, p, b)
                positives_as_dict[f"{b}__{p}"] = positives_filtered   

                negatives_filtered = filter(negatives, variable_suffix, p, b)
                negatives_as_dict[f"{b}__{p}"] = negatives_filtered                                

    return negatives_as_dict, positives_as_dict


def far_threshold(negatives, positives, far_value=0.001, is_sorted=False):


    # N.B.: Unoptimized version ported from C++

    if far_value < 0.0 or far_value > 1.0:
        raise RuntimeError("`far_value' must be in the interval [0.,1.]")

    if len(negatives) < 2:
        raise RuntimeError("the number of negative scores must be at least 2")

    epsilon = numpy.finfo(numpy.float64).eps
    # if not pre-sorted, copies and sorts
    scores = negatives if is_sorted else numpy.sort(negatives)

    # handles special case of far == 1 without any iterating
    if far_value >= (1 - epsilon):
        return numpy.nextafter(scores[0], scores[0] - 1)

    # Reverse negatives so the end is the start. This way the code below will
    # be very similar to the implementation in the frr_threshold function. The
    # implementations are not exactly the same though.
    scores = numpy.flip(scores)

    # Move towards the end of array changing the threshold until we cross the
    # desired FAR value. Starting with a threshold that corresponds to FAR ==
    # 0.
    total_count = len(scores)
    current_position = 0

    # since the comparison is `if score >= threshold then accept as genuine`,
    # we can choose the largest score value + eps as the threshold so that we
    # can get for 0% FAR.
    valid_threshold = numpy.nextafter(
        scores[current_position], scores[current_position] + 1
    )
    current_threshold = 0.0

    while current_position < total_count:
        current_threshold = scores[current_position]
        # keep iterating if values are repeated
        while (
            current_position < (total_count - 1)
            and scores[current_position + 1] == current_threshold
        ):
            current_position += 1
        # All the scores up to the current position and including the current
        # position will be accepted falsely.
        future_far = (current_position + 1) / total_count
        if future_far > far_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


def compute_fmr_thresholds(negatives, fmrs=[0.1, 0.01, 0.001]):

    negatives_as_np = negatives.compute()["score"].to_numpy().astype("float64")
    taus = [far_threshold(negatives_as_np, [], far_value=t) for t in fmrs]

    return taus

def farfrr(negatives, positives, threshold):


    if numpy.isnan(threshold):
        print("Error: Cannot compute FPR (FAR) or FNR (FRR) with NaN threshold")
        return (1.0, 1.0)

    if not len(negatives):
        raise RuntimeError(
            "Cannot compute FPR (FAR) when no negatives are given"
        )

    if not len(positives):
        raise RuntimeError(
            "Cannot compute FNR (FRR) when no positives are given"
        )

    return (negatives >= threshold).sum() / len(negatives), (
        positives < threshold
    ).sum() / len(positives)
    

def compute_werm_geo(A_tau, B_tau, alpha=0.5, beta=0.5, epsilon=1e-5):
    max_A_tau = max(A_tau)
    geo_mean_A_tau = numpy.prod([x + epsilon for x in A_tau]) ** (1 / len(A_tau))
    term_A = (max_A_tau / geo_mean_A_tau) ** alpha

    max_B_tau = max(B_tau)
    geo_mean_B_tau = numpy.prod([x + epsilon for x in B_tau]) ** (1 / len(B_tau))
    term_B = (max_B_tau / geo_mean_B_tau) ** beta

    return term_A, term_B, term_A * term_B
