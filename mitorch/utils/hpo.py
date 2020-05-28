#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

from itertools import product
import warnings


def len_hp_set(hp_set):
    output_len = 1
    for v in hp_set.values():
        output_len *= len(v)

    return output_len


def len_hp_param(hp_param):
    output_len = 1
    for v in hp_param:
        if v['type'] == 'choice':
            output_len *= len(v['values'])

    return output_len


def set_hp_cfg(cfg, in_item):
    key, value = in_item
    assert isinstance(key, str) and len(key)
    key_list = key.split('.')
    key_par = cfg
    for i, k in enumerate(key_list):
        if i == len(key_list) - 1:
            break
        key_par = key_par.get(k, None)
    setattr(key_par, key_list[-1], value)

    return cfg


def hp_gen_set_cfg(hps_tuple, cfg):
    hps_dict = dict()
    for k, v in hps_tuple:
        cfg = set_hp_cfg(cfg, (k, v))
        hps_dict[k] = v

    return hps_dict, cfg


def hp_gen(cfg, hp_set):
    for hps in product(*hp_set.values()):
        hps_tuple = tuple(zip(hp_set.keys(), hps))
        yield hp_gen_set_cfg(hps_tuple, cfg)


def exp_range_finder(cfg, len_exps):
    hpo = cfg.get('HPO')
    hpo_range_start = hpo.get('RANGE_START')
    hpo_range_len = hpo.get('RANGE_LEN')
    hpo_range_len = len_exps if hpo_range_len == 0 else hpo_range_len
    hpo_range_end = hpo_range_start + hpo_range_len
    assert 0 <= hpo_range_start
    assert 0 < hpo_range_len
    if hpo_range_start >= len_exps:
        warnings.warn('hpo_range_start >= len_exps')
    if hpo_range_end > len_exps:
        warnings.warn('hpo_range_end > len_exps')

    return hpo_range_start, hpo_range_end
