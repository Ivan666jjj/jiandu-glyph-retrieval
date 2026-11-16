# -*- coding: utf-8 -*-
"""自适应融合：不预设残缺度，用「补全前后置信度」决定走哪条链路。

规则：
  分别算「残图检索」和「补全后检索」的 top-1 相似度
  取相似度更高的那一路作为最终结果

这样汉简 70% 档（补全有害）会自动退回残图，
秦简 70% 档（补全有益）会自动采用补全结果。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/Users/Admin/Desktop/02_古文字/残字模型')
from fusion_battle import (
    RetNet, UNet, DEV, IMG, to_t, repair, cut_d, MODES_ALL,
    encode, encode_single, _ENC,
)

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'


def search_with_conf(img, EG, gY, gS, slip):
    """返回 (候选列表, top1 相似度)"""
    e = encode_single(img)
    sims = (e @ EG.T)[0].detach().cpu().numpy()
    order = np.argsort(-sims)
    out, seen = [], set()
    for o in order:
        o = int(o)
        if gS[o] == slip:
            continue
        c = gY[o]
        if c in seen:
            continue
        seen.add(c)
        out.append((c, float(sims[o])))
        if len(out) >= 20:
            break
    conf = out[0][1] if out else -1.0
    return out, conf


def rank_of(cands, truth):
    for k, (c, _) in enumerate(cands):
        if c == truth:
            return k + 1
    return None


def run_adaptive(tag, X, Y, SL, tr, te, model_path, chs, gen, gen_ch, out_json):
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    net = RetNet(len(ck['classes']), chs).to(DEV)
    net.load_state_dict(ck['state']); net.eval()
    _ENC['net'] = net

    print(f'\n{tag}：gallery {len(tr)} / 测试 {len(te)}', flush=True)
    EG = encode(net, [X[i] for i in tr])
    gY = [Y[i] for i in tr]; gS = [SL[i] for i in tr]

    res = {}
    for ratio in [0.3, 0.5, 0.7]:
        rng = np.random.default_rng(2029)
        stat = {'raw': {1: 0, 5: 0, 10: 0, 15: 0, 20: 0},
                'fix': {1: 0, 5: 0, 10: 0, 15: 0, 20: 0},
                'ada': {1: 0, 5: 0, 10: 0, 15: 0, 20: 0}}
        choose_fix = 0
        n = 0
        for i in te:
            dmg, mk = cut_d(X[i], rng.choice(MODES_ALL), ratio)
            fixed = repair(gen, dmg, mk)

            c_raw, conf_raw = search_with_conf(dmg, EG, gY, gS, SL[i])
            c_fix, conf_fix = search_with_conf(fixed, EG, gY, gS, SL[i])

            # 自适应：谁置信度高用谁
            if conf_fix > conf_raw:
                chosen = c_fix; choose_fix += 1
            else:
                chosen = c_raw

            for key, cands in [('raw', c_raw), ('fix', c_fix), ('ada', chosen)]:
                r = rank_of(cands, Y[i])
                for k in stat[key]:
                    if r and r <= k:
                        stat[key][k] += 1
            n += 1

        row = {}
        print(f'\n  裁切 {int(ratio*100)}%  （自适应选中补全的占比 {choose_fix/n*100:.0f}%）')
        print(f'    {"链路":<12}{"R1":<8}{"R5":<8}{"R10":<8}{"R15":<8}{"R20":<8}')
        for key, label in [('raw', '残图直接'), ('fix', '补全→检索'), ('ada', '自适应')]:
            v = {k: stat[key][k] / n * 100 for k in stat[key]}
            row[key] = v
            print(f'    {label:<12}{v[1]:<8.1f}{v[5]:<8.1f}{v[10]:<8.1f}{v[15]:<8.1f}{v[20]:<8.1f}')
        res[f'{int(ratio*100)}%'] = row

    import json
    json.dump(res, open(out_json, 'w'), ensure_ascii=False, indent=1)
    print(f'\n  已存 {out_json}', flush=True)
    return res


if __name__ == '__main__':
    import collections, json

    which = sys.argv[1] if len(sys.argv) > 1 else 'han'
    if which == 'han':
        d = np.load(os.path.join(HERE, 'dahan1/chars.npz'), allow_pickle=True)
        X = np.array(d['X']); Y = np.array([str(v) for v in d['y']])
        SL = np.array([str(v) for v in d['xml']]); SP = np.array([str(v) for v in d['split']])
        tr = np.where(SP == 'train')[0]; te = np.where(SP == 'test')[0]
        cnt = collections.Counter(Y[tr]); keep = {c for c, n in cnt.items() if n >= 2}
        tr = np.array([i for i in tr if Y[i] in keep])
        te = np.array([i for i in te if Y[i] in keep])[:600]

        ckg = torch.load(os.path.join(HERE, 'restorer_cloud/restorer_direct.pt'),
                         map_location='cpu', weights_only=False)
        gen = UNet().to(DEV); gen.load_state_dict(ckg['state']); gen.eval()

        run_adaptive('大汉七号 + 自适应补全', X, Y, SL, tr, te,
                     os.path.join(HERE, 'xh7_final/dahan7.pt'),
                     (32, 64, 128, 256), gen, (64, 128, 256, 512),
                     '/tmp/ada_han.json')
    else:
        d = np.load(os.path.join(HERE, 'beida/beida_labeled.npz'), allow_pickle=True)
        X = np.array(d['X']); Y = np.array([str(v) for v in d['y']])
        SL = np.array([str(v) for v in d['slip']])
        sj = json.load(open(os.path.join(HERE, 'beida/split.json'), encoding='utf-8'))
        sp = np.array(['train'] * len(X))
        for it in sj['items']:
            i = int(it['idx'])
            if i < len(sp):
                sp[i] = it['split']
        tr = np.where(sp == 'train')[0]; te = np.where(sp == 'test')[0]
        cnt = collections.Counter(Y[tr]); keep = {c for c, n in cnt.items() if n >= 2}
        tr = np.array([i for i in tr if Y[i] in keep])
        te = np.array([i for i in te if Y[i] in keep])

        ckg = torch.load(os.path.join(HERE, 'restorer_v3/qin/restorer_v3_small.pt'),
                         map_location='cpu', weights_only=False)
        gen = UNet((32, 64, 128, 256)).to(DEV)
        gen.load_state_dict(ckg['state']); gen.eval()

        run_adaptive('小焕七号 + 自适应补全', X, Y, SL, tr, te,
                     os.path.join(HERE, 'xh7_final/xh7_C_全量+字典.pt'),
                     (32, 64, 128, 256), gen, (32, 64, 128, 256),
                     '/tmp/ada_qin.json')
