from __future__ import division
import nltk
from collections import Counter

WORDS = nltk.corpus.brown.words()
COUNTS = Counter(WORDS)


def dis(counter):
    N = sum(counter.values())
    return lambda x: counter[x]/N


k = dis(COUNTS)

def wo(wo):
    return multi(k(w) for w in wo)


def multi(nums):
    result = 1
    for x in nums:
        result *= x
    return result


def splits(text, start=0, L=20):
    if not text:
        return []
    else:
        candidates = ([first] + segment(rest)
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=wo)


def segment(start , L ):
    return [(start[:i], L[i:])
            for i in range(start, min(len(start), L)+1)]
    