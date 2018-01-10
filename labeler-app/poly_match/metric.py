from sys import path
path.append('..')
path.append('./build')

import utm
import json
import argparse
import numpy as np
from modeling import Model
# from poly_match import find_affine, OptimizationParamsBuilder
from affine import AffineMx

def parse_params(args):
    builder = OptimizationParamsBuilder()

    if min_shift is not None:
        builder.set_min_shift(args.min_shift)

    if max_shift is not None:
        builder.set_max_shift(args.max_shift)

    if min_theta is not None:
        builder.set_min_theta(args.min_theta)

    if max_theta is not None:
        builder.set_max_theta(args.max_theta)

    if min_scale is not None:
        builder.set_min_scale(args.min_scale)

    if max_scale is not None:
        builder.set_max_scale(args.max_scale)

    if grid_step is not None:
        builder.set_grid_step(args.grid_step)

    if desc_num_steps is not None:
        builder.set_desc_num_steps(args.desc_num_steps)
    if desc_lr is not None:
        builder.set_learn_rate(args.learn_rate)

    params = builder.build()

    return params

def main(args):
    with open(args.real, 'r') as f:
        real_polys = json.load(f)

    with open(args.gen, 'r') as f:
        pred_polys = json.load(f)

    assert(len(real_polys) == len(pred_polys))

    opt_params = parse_params(args)
    for real, pred in zip(real_polys, pred_polys):
        transform = find_affine(Polygon(real), Polygon(pred), opt_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True, help='path to json file with list polygons '
                                                      '[poly1, poly2, poly3]')
    parser.add_argument('--gen', required=True, help='path to json file with list generated polygons '
                                                     '[gen_poly1, gen_poly2, gen_poly3]'
                                                     'gen_poly1 will be compared with poly1 etc')
    parser.add_argument('--out', required=True, help='output file name')

    grid_group = parser.add_argument_group('optimization parameters for grid search')
    grid_group.add_argument('--min_shift', type=float)
    grid_group.add_argument('--max_shift', type=float)
    grid_group.add_argument('--min_theta', type=float)
    grid_group.add_argument('--max_theta', type=float)
    grid_group.add_argument('--min_scale', type=float)
    grid_group.add_argument('--max_scale', type=float)
    grid_group.add_argument('--grid_step', type=float)

    grad_group = parser.add_argument_group('optimization parameters for gradient descent')
    grad_group.add_argument('--desc_num_steps', type=float)
    grad_group.add_argument('--learn_rate', type=float)

    args = parser.parse_args()
