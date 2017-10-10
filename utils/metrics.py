from math import log

def entropy(Y):
    """
    Y: target variable of size (n,) (1 dim)
    """
    Y_num = np.unique(Y).shape[0]
    Y_prob = np.histogram(Y, bins=Y_num, density=True)[0]
    return (-1)*np.sum([p*np.log2(p) for p in Y_prob])

def elog(x):
    # for entropy, 0 log 0 = 0. but we get an error for putting log 0
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x * log(x)

def hist(sx):
    # Histogram from list of samples
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z) / len(sx), d.values())

def entropyfromprobs(probs, base=2):
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return -sum(map(elog, probs)) / log(base)

def entropyd(sx, base=2):
    """
    Discrete entropy estimator given a list of samples which can be any hashable object
    """
    # import pdb; pdb.set_trace()
    return entropyfromprobs(hist(sx), base=base)

def midd(x, y):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """
    return -entropyd(list(zip(x, y))) + entropyd(x) + entropyd(y)

def rig(X, Y):
    pe = entropyd(Y)
    ce = entropyd(Y) - midd(X, Y)
    return (pe - ce) / float(pe)