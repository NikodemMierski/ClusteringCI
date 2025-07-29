"""
This module is adapted from code originally found at:
https://github.com/psinger/RunsTest

Author: Peter Singer 

No license was provided in the original repository.
"""

__author__ = 'psinger'

from collections import defaultdict
import math
from scipy.stats import chi2

def weighted_variance(counts):
    avg = 0
    for length, count in counts.items():
        avg += count * length

    counts_only = list(counts.values())
    avg /= sum(counts_only)

    var = 0
    for length, count in counts.items():
        var += count * math.pow((length - avg), 2)

    try:
        var /= sum(counts_only) - 1
    except ZeroDivisionError:
        raise Exception("Division by zero due to too few counts!")

    return var

def runs_test(input_data, path=True):
    '''
    You can pass a path or a dictionary of run lengths.
    If path=True, treat input_data as a sequence; else as a precomputed dict.
    '''
    if path:
        counter = 1
        cats = defaultdict(lambda: defaultdict(int))

        for i, elem in enumerate(input_data):
            if i == len(input_data) - 1:
                cats[elem][counter] += 1
                break

            if input_data[i + 1] == elem:
                counter += 1
            else:
                cats[elem][counter] += 1
                counter = 1
    else:
        cats = input_data

    x2 = 0
    df = 0
    nr_elem = len(cats.keys())
    fail_cnt = 0

    for elem in cats.keys():
        ns = sum([x * y for x, y in cats[elem].items()])
        rs = sum(cats[elem].values())

        if len(cats[elem].keys()) == 1 or rs == 1 or (ns - rs) == 1:
            fail_cnt += 1
            continue

        ss = weighted_variance(cats[elem])
        cs = ((rs ** 2) - 1) * (rs + 2) * (rs + 3) / (2 * rs * (ns - rs - 1) * (ns + 1))
        vs = cs * ns * (ns - rs) / (rs * (rs + 1))

        x2 += ss * cs
        df += vs

    if nr_elem - fail_cnt < 2:
        raise Exception("Too many categories were ignored. Cannot perform the test.")

    if x2 == 0 or df == 0:
        raise Exception("x2 or df is zero, cannot compute p-value.")

    pval = chi2.sf(x2, df)
    return pval