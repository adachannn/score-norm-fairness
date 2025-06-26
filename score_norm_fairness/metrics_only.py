import numpy
import torch
from scipy.special import betaincinv
import numpy as np

def split_scores_by_variable(negatives, positives, variable_suffix):
    demos = negatives[variable_suffix].unique()

    negatives_as_dict = dict()
    positives_as_dict = dict()

    def filter(df, variable_suffix, pair_demo):
        return df.loc[df[variable_suffix] == pair_demo]

    for b in demos:
        positives_filtered = filter(positives, variable_suffix, b)
        positives_as_dict[b] = positives_filtered   

        negatives_filtered = filter(negatives, variable_suffix, b)
        negatives_as_dict[b] = negatives_filtered                                

    return negatives_as_dict, positives_as_dict

def far_threshold(negatives, positives, far_value=0.001, is_sorted=False):

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


def compute_fmr_thresholds(negatives, fmrs=[0.001]):
    taus = [far_threshold(negatives, [], far_value=t) for t in fmrs]

    return taus[0]

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

    false_positive = (negatives >= threshold).sum()
    false_negative = (positives < threshold).sum()
    far = false_positive / len(negatives)
    frr = false_negative / len(positives)
    return false_positive, false_negative, far, frr

def compute_differential_error(A_tau, B_tau):
    term_A = max(A_tau) - min(A_tau)
    term_B = max(B_tau) - min(B_tau)
    return term_A, term_B

def compute_ser(A_tau, B_tau, epsilon_a=1e-6, epsilon_b=1e-6):
    min_A = min(A_tau)
    max_A = max(A_tau)
    min_B = min(B_tau)
    max_B = max(B_tau)
    term_A = max(A_tau) / (min(A_tau) + epsilon_a)
    term_B = max(B_tau) / (min(B_tau) + epsilon_b)
    return term_A, term_B

def compute_merg(A_tau, B_tau, alpha=0.5, beta=0.5, epsilon_a=1e-6, epsilon_b=1e-6):
    max_A_tau = max(A_tau)
    geo_mean_A_tau = numpy.prod([x + epsilon_a for x in A_tau]) ** (1 / len(A_tau))
    term_A = (max_A_tau / geo_mean_A_tau) ** alpha

    max_B_tau = max(B_tau)
    geo_mean_B_tau = numpy.prod([x + epsilon_b for x in B_tau]) ** (1 / len(B_tau))
    term_B = (max_B_tau / geo_mean_B_tau) ** beta

    return term_A, term_B

def compute_mera(A_tau, B_tau, alpha=0.5, beta=0.5):
    max_A_tau = max(A_tau)
    mean_A_tau = numpy.mean(A_tau)
    term_A = (max_A_tau / mean_A_tau) ** alpha

    max_B_tau = max(B_tau)
    mean_B_tau = numpy.mean(B_tau)
    term_B = (max_B_tau / mean_B_tau) ** beta

    return term_A, term_B


def compute_gini(A_tau, B_tau, alpha=0.5, beta=0.5):
    numerator_A_tau = sum([abs(x - y) for x in A_tau for y in A_tau])
    numerator_B_tau = sum([abs(x - y) for x in B_tau for y in B_tau])
    mean_A_tau = numpy.mean(A_tau)
    mean_B_tau = numpy.mean(B_tau)
    denominator_A_tau = 2 * len(A_tau) * (len(A_tau) - 1) * mean_A_tau
    denominator_B_tau = 2 * len(B_tau) * (len(B_tau) - 1) * mean_B_tau
    term_A = numerator_A_tau / denominator_A_tau
    term_B = numerator_B_tau / denominator_B_tau

    return term_A, term_B


def upper_bound_error_rate(x, n, alpha=0.05):
    """
    Calculates the upper bound on the error rate for a given number of errors x, 
    total trials n, and confidence level alpha.

    Parameters:
    x (int): Observed number of errors.
    n (int): Total number of trials.
    alpha (float): Confidence level (default is 0.05 for 95% confidence).

    Returns:
    float: Upper bound on the error rate.
    """
    
    # Calculate the upper bound of the error rate using the inverse of the incomplete beta function
    return betaincinv(x + 1, n - x, 1-alpha)


def zero_error_rate(x, n, alpha=0.05):

    if x == 0:
    # Calculate the upper bound of the error rate using the inverse of the incomplete beta function
        return betaincinv(x + 1, n - x, 1-alpha)
    else:
        return x/n

def compute_merg_adjusted(A_tau, B_tau, alpha=0.5, beta=0.5):

    term_A = numpy.max(numpy.array(A_tau[0])) / numpy.exp(numpy.mean(numpy.log(numpy.array(A_tau[1]))))
    term_B = numpy.max(numpy.array(B_tau[0])) / numpy.exp(numpy.mean(numpy.log(numpy.array(B_tau[1]))))

    return term_A, term_B

def compute_inequity(term_A, term_B, alpha=0.5, beta=0.5):
    """
    Computes the combined FMR and FNMR inequity.

    Parameters:
    - term_A (float): FMR term.
    - term_B (float): FNMR term.
    - alpha (float): Weighting factor for FMR (default is 0.5).
    - beta (float): Weighting factor for FNMR (default is 0.5).

    Returns:
    - float: Combined inequity value.
    """
    return (alpha * term_A) + (beta * term_B)

### Below is the code for equal error rate computation, as you can see from the given papers
def abs_diff(a, b, cost):
    return abs(a - b)

def weighted_err(far, frr, cost):
    return (cost * far) + ((1.0 - cost) * frr)

def minimizing_threshold(negatives, positives, criterion, cost=0.5):
    """Calculates the best threshold taking a predicate as input condition

    This method can calculate a threshold based on a set of scores (positives
    and negatives) given a certain minimization criterium, input as a
    functional predicate. For a discussion on ``positive`` and ``negative`` see
    :py:func:`farfrr`.  Here, it is expected that the positives and the
    negatives are sorted ascendantly.

    The predicate method gives back the current minimum given false-acceptance
    (FA) and false-rejection (FR) ratios for the input data. The API for the
    criterium is:

    predicate(fa_ratio : float, fr_ratio : float) -> float

    Please note that this method will only work with single-minimum smooth
    predicates.

    The minimization is carried out in a data-driven way.  Starting from the
    lowest score (might be a positive or a negative), it increases the
    threshold based on the distance between the current score and the following
    higher score (also keeping track of duplicate scores) and computes the
    predicate for each possible threshold.

    Finally, that threshold is returned, for which the predicate returned the
    lowest value.


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)
        Negative scores, sorted ascendantly

    positives : numpy.ndarray (1D, float)
        Positive scores, sorted ascendantly

    criterion : str
        A predicate from one of ("absolute-difference", "weighted-error")

    cost : float
        Extra cost argument to be passed to criterion

    Returns
    =======

    threshold : float
        The optimal threshold given the predicate and the scores

    """

    if criterion not in ("absolute-difference", "weighted-error"):
        raise ValueError("Uknown criterion")

    def criterium(a, b, c):
        if criterion == "absolute-difference":
            return abs_diff(a, b, c)
        else:
            return weighted_err(a, b, c)

    if not len(negatives) or not len(positives):
        raise RuntimeError(
            "Cannot compute threshold when no positives or "
            "no negatives are provided"
        )

    # iterates over all possible far and frr points and compute the predicate
    # for each possible threshold...
    min_predicate = 1e8
    min_threshold = 1e8
    current_predicate = 1e8
    # we start with the extreme values for far and frr
    far = 1.0
    frr = 0.0

    # the decrease/increase for far/frr when moving one negative/positive
    max_neg = len(negatives)
    far_decrease = 1.0 / max_neg
    max_pos = len(positives)
    frr_increase = 1.0 / max_pos

    # we start with the threshold based on the minimum score

    # iterates until one of these goes bananas
    pos_it = 0
    neg_it = 0
    current_threshold = min(negatives[neg_it], positives[pos_it])

    # continues until one of the two iterators reaches the end of the list
    while pos_it < max_pos and neg_it < max_neg:
        # compute predicate
        current_predicate = criterium(far, frr, cost)

        if current_predicate <= min_predicate:
            min_predicate = current_predicate
            min_threshold = current_threshold

        if positives[pos_it] >= negatives[neg_it]:
            # compute current threshold
            current_threshold = negatives[neg_it]
            neg_it += 1
            far -= far_decrease

        else:  # pos_val <= neg_val
            # compute current threshold
            current_threshold = positives[pos_it]
            pos_it += 1
            frr += frr_increase

        # skip until next "different" value, which case we "gain" 1 unit on
        # the "FAR" value, since we will be accepting that negative as a
        # true negative, and not as a false positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while neg_it < max_neg and current_threshold == negatives[neg_it]:
            neg_it += 1
            far -= far_decrease

        # skip until next "different" value, which case we "loose" 1 unit
        # on the "FRR" value, since we will be accepting that positive as a
        # false negative, and not as a true positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while pos_it < max_pos and current_threshold == positives[pos_it]:
            pos_it += 1
            frr += frr_increase

        # computes a new threshold based on the center between last and current
        # score, if we are **not** already at the end of the score lists
        if neg_it < max_neg or pos_it < max_pos:
            if neg_it < max_neg and pos_it < max_pos:
                current_threshold += min(negatives[neg_it], positives[pos_it])
            elif neg_it < max_neg:
                current_threshold += negatives[neg_it]
            else:
                current_threshold += positives[pos_it]
            current_threshold /= 2

    # now, we have reached the end of one list (usually the negatives) so,
    # finally compute predicate for the last time

    current_predicate = criterium(far, frr, cost)
    if current_predicate < min_predicate:
        min_predicate = current_predicate
        min_threshold = current_threshold

    # now we just double check choosing the threshold higher than all scores
    # will not improve the min_predicate
    if neg_it < max_neg or pos_it < max_pos:
        last_threshold = current_threshold
        if neg_it < max_neg:
            last_threshold = numpy.nextafter(negatives[-1], negatives[-1] + 1)
        elif pos_it < max_pos:
            last_threshold = numpy.nextafter(positives[-1], positives[-1] + 1)
        current_predicate = criterium(0.0, 1.0, cost)
        if current_predicate < min_predicate:
            min_predicate = current_predicate
            min_threshold = last_threshold

    # return the best threshold found
    return min_threshold

def eer_threshold(negatives, positives, is_sorted=False):
    """Calculates threshold as close as possible to the equal error rate (EER)

    The EER should be the point where the FPR equals the FNR. Graphically, this
    would be equivalent to the intersection between the ROC (or DET) curves and
    the identity.

    .. note::

       The scores will be sorted internally, requiring the scores to be copied.
       To avoid this copy, you can sort both sets of scores externally in
       ascendant order, and set the ``is_sorted`` parameter to ``True``.


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)

        The set of negative scores to compute the threshold

    positives : numpy.ndarray (1D, float)

        The set of positive scores to compute the threshold

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``negatives`` are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold (i.e., as used in :py:func:`farfrr`) where FPR and FNR
        are as close as possible

    """

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)

    return minimizing_threshold(neg, pos, "absolute-difference")

def eer(negatives, positives, is_sorted=False, also_farfrr=False):
  """Calculates the Equal Error Rate (EER).

  Please note that it is possible that eer != fpr != fnr.
  This function returns (fpr + fnr) / 2 as eer.
  If you also need the fpr and fnr values, set ``also_farfrr`` to ``True``.

  Parameters
  ----------
  negatives : ``array_like (1D, float)``
      The scores for comparisons of objects of different classes.
  positives : ``array_like (1D, float)``
      The scores for comparisons of objects of the same class.
  is_sorted : bool
      Are both sets of scores already in ascendantly sorted order?
  also_farfrr : bool
      If True, it will also return far and frr.

  Returns
  -------
  eer : float
      The Equal Error Rate (EER).
  fpr : float
      The False Positive Rate (FPR). Returned only when ``also_farfrr`` is
      ``True``.
  fnr : float
      The False Negative Rate (FNR). Returned only when ``also_farfrr`` is
      ``True``.
  """
  threshold = eer_threshold(negatives, positives, is_sorted)
  _, _, far, frr = farfrr(negatives, positives, threshold)
  if also_farfrr:
    return (far + frr) / 2.0, far, frr, threshold
  return (far + frr) / 2.0, threshold

def compute_accuracy(score_vec, label_vec, threshold=None):
    assert len(score_vec.shape)==1
    assert len(label_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == int or label_vec.dtype == np.int32 or label_vec.dtype == np.int64
    
    # find thresholds by TAR
    if threshold is None:
        score_pos = score_vec[label_vec==1]
        thresholds = np.sort(score_pos)[::1]    
        if np.size(thresholds) > 10000:
            print('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))    
        # Loop Computation
        accuracies = np.zeros(np.size(thresholds))
        for i, threshold in enumerate(thresholds):
            pred_vec = score_vec>=threshold
            accuracies[i] = np.mean(pred_vec==label_vec)
        # Matrix Computation, Each column is a threshold
        argmax = np.argmax(accuracies)
        accuracy = accuracies[argmax]
        threshold = np.mean(thresholds[accuracies==accuracy])
    else:
        pred_vec = score_vec>=threshold
        accuracy = np.mean(pred_vec==label_vec)

    return accuracy, threshold
