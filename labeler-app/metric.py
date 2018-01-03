import numpy as np
from modeling import Model
from polygon_matching import PolyMatcher
from affine import AffineMx


def perplexity(model, poly_pairs):
    """
    :param model:
    :param poly_pairs:
    :return:
    """
    xs = []

    for poly_real, poly_pred in poly_pairs:
        affine_trans = PolyMatcher.find_best_affine_match(
            poly_real, poly_pred
        )

        mx = AffineMx.trans_from_params(affine_trans["shift"],
                                        affine_trans["theta"],
                                        affine_trans["scale"])
        x = np.hstack((mx, affine_trans["distance"]))
        xs.append(x)

    model.predict_probas(np.array(xs))


if __name__ == '__main__':
    pass
