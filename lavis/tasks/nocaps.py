"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("nocaps")
class NoCapsTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        urls = {
            "val": "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
        }
        filenames = {
            "val": "nocaps_val_4500_captions.json",
        }

        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        download_url(urls[split_name], coco_gt_root)
        annotation_file = os.path.join(coco_gt_root, filenames[split_name])
        annotations = json.load(open(annotation_file))
        images = annotations["images"]
        id2domain = {im['id']: im['domain'] for im in images}

        # file split to 3 splits
        to_split_files = {sp: {"images": [], "annotations": []} for sp in ["in-domain", "near-domain", "out-domain"]}
        for im in images:
            to_split_files[im["domain"]]["images"].append(im)
        for ann in annotations["annotations"]:
            to_split_files[id2domain[ann["image_id"]]]["annotations"].append(ann)
        for sp, v in to_split_files.items():
            fname = os.path.join(coco_gt_root, filenames[split_name]+f".{sp}")
            json.dump(v, open(fname, "w"))

        all_eval_resuls = {"in-domain": [], "near-domain": [], "out-domain":[], "all": val_result}
        for k in val_result:
            all_eval_resuls[id2domain[k['image_id']]].append(k)
        for k, v in all_eval_resuls.items():
            eval_result_file = self.save_result(
                result=v,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}_{}".format(split_name, epoch, k),
                remove_duplicate="image_id",
            )

            if self.report_metric:
                metrics = self._report_metrics(
                    eval_result_file=eval_result_file, split_name=f"{split_name}_{k}"
                )
            else:
                metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val_in-domain": "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
        "val_near-domain": "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
        "val_out-domain": "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
        "val_all": "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
    }
    filenames = {
        "val_in-domain": "nocaps_val_4500_captions.json.in-domain",
        "val_near-domain": "nocaps_val_4500_captions.json.near-domain",
        "val_out-domain": "nocaps_val_4500_captions.json.out-domain",
        "val_all": "nocaps_val_4500_captions.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
