from sklearn.metrics import fbeta_score
import numpy as np

def getScore(y,p,x):
    p2 = np.zeros_like(p)
    for i in range(17):
        p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    x = [0.2]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = getScore(y,p,x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)
    return x

# def optimise_f2_thresholds(y, p, verbose=True, resolution=[100] * 17, base_x=None):
#     if base_x is None:
#         x = [0.2] * 17
#     else:
#         x = base_x
#     p2 = (p > x).astype(np.int)
#     _tp = ((p2 == 1) & (p2 == y)).astype('int')
#     _fp = ((p2 == 1) & (p2 != y)).astype('int')
#     _fn = ((p2 == 0) & (p2 != y)).astype('int')
#     _tp_sum = np.sum(_tp, 1)
#     _fp_sum = np.sum(_fp, 1)
#     _fn_sum = np.sum(_fn, 1)
#
#     for i in range(17):
#         best_i2 = 0
#         best_score = 0
#         score_range = []
#         for i2 in range(resolution[i]):
#             i2 /= 1.0 * resolution[i]
#             x[i] = i2
#             tmp = (p[:, i] > x[i]).astype(np.int)
#             if np.all(tmp == p2[:, i]):
#                 continue
#             else:
#                 p2[:, i] = tmp
#             _tp_sum = _tp_sum - _tp[:, i] + ((tmp == 1) & (tmp == y[:, i])).astype('int')
#             _fp_sum = _fp_sum - _fp[:, i] + ((tmp == 1) & (tmp != y[:, i])).astype('int')
#             _fn_sum = _fn_sum - _fn[:, i] + ((tmp == 0) & (tmp != y[:, i])).astype('int')
#             _tp[:, i] = ((tmp == 1) & (tmp == y[:, i])).astype('int')
#             _fp[:, i] = ((tmp == 1) & (tmp != y[:, i])).astype('int')
#             _fn[:, i] = ((tmp == 0) & (tmp != y[:, i])).astype('int')
#             _p = 1.0 * _tp_sum / (_tp_sum + _fp_sum)
#             _r = 1.0 * _tp_sum / (_tp_sum + _fn_sum)
#             score = np.mean(5.0 * _p * _r / (np.maximum(4 * _p + _r, 1e-15)))
#             if score > best_score:
#                 best_i2 = i2
#                 best_score = score
#         x[i] = best_i2
#         tmp = (p[:, i] > x[i]).astype(np.int)
#         if np.all(tmp == p2[:, i]):
#             continue
#         else:
#             p2[:, i] = tmp
#         _tp_sum = _tp_sum - _tp[:, i] + ((tmp == 1) & (tmp == y[:, i])).astype('int')
#         _fp_sum = _fp_sum - _fp[:, i] + ((tmp == 1) & (tmp != y[:, i])).astype('int')
#         _fn_sum = _fn_sum - _fn[:, i] + ((tmp == 0) & (tmp != y[:, i])).astype('int')
#         _tp[:, i] = ((tmp == 1) & (tmp == y[:, i])).astype('int')
#         _fp[:, i] = ((tmp == 1) & (tmp != y[:, i])).astype('int')
#         _fn[:, i] = ((tmp == 0) & (tmp != y[:, i])).astype('int')
#         p2[:, i] = tmp
#
#         if verbose:
#             print(i, best_i2, best_score)
#     print('Train', best_score)
#     return x