import argparse
import copy
import json
import os
import os.path as osp
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from hawp.base import WireframeGraph, show
from hawp.fsl.config import cfg as model_config
from hawp.ssl.models import MODELS

_MODEL_CACHE = {}

def _build_parser():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--metarch', default='HAWP', choices=MODELS.keys())
    aparser.add_argument(
        '--cfg',
        default=os.path.join(os.path.dirname(__file__), 'config', 'hawpv3.yaml'),
        help='model configuration',
    )
    aparser.add_argument('--ckpt', required=True, help='checkpoint')
    aparser.add_argument('-t', '--threshold', default=0.05, type=float)
    aparser.add_argument('--img', required=True, type=str, nargs='+')
    aparser.add_argument('--width', default=512, type=int)
    aparser.add_argument('--height', default=512, type=int)
    aparser.add_argument('--whitebg', default=0.0, type=float)
    aparser.add_argument('--saveto', default=None, type=str)
    aparser.add_argument('--ext', default='pdf', type=str, choices=['pdf', 'png', 'json', 'txt'])
    aparser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu', 'mps'])
    aparser.add_argument('--disable-show', default=False, action='store_true')

    for k in MODELS.keys():
        MODELS[k].cli(aparser)

    return aparser


def parse_args():
    aparser = _build_parser()
    args = aparser.parse_args()

    for k in MODELS.keys():
        MODELS[k].configure(args)
    show.painters.HAWPainter.confidence_threshold = args.threshold

    return args


def _load_model(args):
    cache_key = (args.metarch, args.cfg, args.ckpt, args.device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cfg = model_config.clone()
    cfg.merge_from_file(args.cfg)
    model = MODELS[args.metarch](cfg, gray_scale=True)
    model = model.eval().to(args.device)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    _MODEL_CACHE[cache_key] = model
    return model

def _configured_args(**overrides):
    parser = _build_parser()
    args = parser.parse_args(['--ckpt', '__bundle__.pth', '--img', '__bundle__.png'])
    defaults = {
        'metarch': 'HAWP',
        'cfg': os.path.join(os.path.dirname(__file__), 'config', 'hawpv3.yaml'),
        'ckpt': None,
        'threshold': 0.05,
        'img': [],
        'width': 512,
        'height': 512,
        'whitebg': 0.0,
        'saveto': None,
        'ext': 'json',
        'device': 'cuda',
        'disable_show': True,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(args, key, value)
    for key in MODELS.keys():
        MODELS[key].configure(args)
    show.painters.HAWPainter.confidence_threshold = args.threshold
    return args


def run_prediction(
    *,
    images,
    ckpt,
    threshold,
    width,
    height,
    saveto,
    device,
    ext='json',
    disable_show=True,
    metarch='HAWP',
    cfg=None,
):
    args = _configured_args(
        metarch=metarch,
        cfg=cfg or os.path.join(os.path.dirname(__file__), 'config', 'hawpv3.yaml'),
        ckpt=str(ckpt),
        threshold=float(threshold),
        img=[str(item) for item in images],
        width=int(width),
        height=int(height),
        saveto=str(saveto) if saveto is not None else None,
        ext=ext,
        device=device,
        disable_show=bool(disable_show),
    )
    model = _load_model(args)

    height = args.height
    width = args.width

    if args.saveto:
        os.makedirs(args.saveto, exist_ok=True)
    show.Canvas.show = not args.disable_show
    painter = show.painters.HAWPainter()

    image_list = [cv2.imread(fname, 0) for fname in args.img]
    generated = []
    for fname, image in zip(tqdm(args.img), image_list):
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {fname}")
        pname = Path(fname)
        ori_shape = image.shape[:2]
        image_cp = copy.deepcopy(image)
        image_ = cv2.resize(image_cp, (width, height))
        image_ = torch.from_numpy(image_).float() / 255.0
        image_ = image_[None, None].to(args.device)

        meta = {
            'width': ori_shape[1],
            'height': ori_shape[0],
            'filename': '',
        }

        with torch.no_grad():
            outputs, _ = model(image_, [meta])

        fig_file = None if args.saveto is None or args.ext in ['txt', 'json'] else osp.join(
            args.saveto, pname.with_suffix('.' + args.ext).name
        )

        with show.image_canvas(fname, fig_file=fig_file) as ax:
            painter.draw_wireframe(ax, outputs)

        indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'], outputs['lines_pred'])
        wireframe = WireframeGraph(
            outputs['juncs_pred'],
            outputs['juncs_score'],
            indices,
            outputs['lines_score'],
            outputs['width'],
            outputs['height'],
        )

        if args.saveto is not None and args.ext == 'json':
            outpath = osp.join(args.saveto, pname.with_suffix('.json').name)
            with open(outpath, 'w', encoding='utf-8') as handle:
                json.dump(wireframe.jsonize(), handle)
            generated.append(Path(outpath))
        elif fig_file is not None:
            generated.append(Path(fig_file))

    return generated


def main():
    args = parse_args()
    run_prediction(
        images=args.img,
        ckpt=args.ckpt,
        threshold=args.threshold,
        width=args.width,
        height=args.height,
        saveto=args.saveto,
        device=args.device,
        ext=args.ext,
        disable_show=args.disable_show,
        metarch=args.metarch,
        cfg=args.cfg,
    )


if __name__ == "__main__":
    main()
