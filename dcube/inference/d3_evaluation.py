import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Union
import json
import argparse
import pandas as pd
from collections import defaultdict
from d_cube import D3
import numpy as np


def map_visualcat_supercategory(json_path, annot_path, eval_mode="full"):
    """
    Merges visual category annotations into COCO-format category annotations.

    Parameters:
        json_path (str): Path to COCO annotation JSONs.
        annot_path (str): Path to CSV file with visual categories.
        eval_mode (str): One of {"full", "abs", "pres"}.
    """
    assert eval_mode in ("full", "abs", "pres"), "Invalid eval_mode"

    # Load ground truth annotations
    mode_path = f"{json_path}/d3_{eval_mode}_annotations.json"
    with open(mode_path, "r") as file:
        gt_annotations = json.load(file)

    # Load dataset with visual categories
    annotated_df = pd.read_csv(annot_path, sep=";")
    annotated_df.columns = annotated_df.columns.str.strip()

    # Extract visual categories per ID
    category_cols = ["category_1", "category_2", "category_3"]
    annotated_df[category_cols] = annotated_df[category_cols].astype(str)

    visual_mapping = [
        {
            "id": idx + 1,
            "visual_cats": [row[col] for col in category_cols if row[col] != "nan"],
        }
        for idx, row in annotated_df.iterrows()
    ]

    # Merge visual category info into COCO categories
    categories_annots = gt_annotations["categories"]
    visual_map_by_id = {entry["id"]: entry for entry in visual_mapping}

    for i, cat in enumerate(categories_annots):
        vis = visual_map_by_id.get(cat["id"])
        if vis:
            categories_annots[i] = cat | vis

    # Save the updated annotations
    SAVE_PATH = f"{json_path}/d3_{eval_mode}_annotations_updated.json"
    with open(SAVE_PATH, "w") as f:
        json.dump(gt_annotations, f, indent=2)

    print(f"Updated annotations for {eval_mode} saved to: {SAVE_PATH}")


def run_category_mapping(json_path, annot_path):

    for eval_mode in ["full", "abs", "pres"]:
        output_path = f"{json_path}/d3_{eval_mode}_annotations_updated.json"

        if os.path.exists(output_path):
            print(f"File already exists: {output_path}, skipping processing")
            return

        # Run the mapping function
        map_visualcat_supercategory(json_path, annot_path, eval_mode)


# -------------------------Category-level Evaluation Function for D3-------------------------


def complete_evaluation(pred_path, JSON_ANNOT_PATH, mode="full"):
    """
    Evaluate predictions on the D3 dataset.
    Args:
        pred_path (str): Path to the predictions JSON file.
        JSON_ANNOT_PATH (str): Path to the ground truth annotations directory.
        mode (str): Evaluation mode - "full", "pres", or "abs".

    Returns:
        None: Prints evaluation results to the console.
    """
    assert mode in ("full", "pres", "abs")
    if mode == "full":
        JSON_ANNOT_PATH = os.path.join(JSON_ANNOT_PATH, "d3_full_annotations.json")
    elif mode == "pres":
        JSON_ANNOT_PATH = os.path.join(JSON_ANNOT_PATH, "d3_pres_annotations.json")
    else:
        JSON_ANNOT_PATH = os.path.join(JSON_ANNOT_PATH, "d3_abs_annotations.json")

    # Load ground truth and predictions using COCO API
    coco = COCO(JSON_ANNOT_PATH)
    d3_res = coco.loadRes(pred_path)
    cocoEval = COCOeval(coco, d3_res, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def filter_by_supercategory(coco, supercat_name, mode="full"):
    """
    Filter annotations based on supercategory.

    Args:
        coco: COCO object
        supercat_name: Name of the supercategory to filter by. (Spatial, object, attribute, action, interaction, appearance, and state.)

    Returns:
        img_ids: List of image IDs containing annotations from this supercategory
        ann_ids: List of annotation IDs belonging to this supercategory
    """
    assert mode in ("full", "abs", "pres"), "Invalid eval_mode"

    # Find all categories that contain the specified supercategory
    cat_ids_with_supercat = []
    for cat in coco.dataset["categories"]:
        if supercat_name in cat.get("visual_cats", []):
            cat_ids_with_supercat.append(cat["id"])

    # print(f"Found {len(cat_ids_with_supercat)} categories/expressions with supercategory '{supercat_name}'")

    # Extract annotations belonging to these categories
    ann_ids = []
    img_ids = set()

    for ann in coco.dataset["annotations"]:
        if ann["category_id"] in cat_ids_with_supercat:
            ann_ids.append(ann["id"])
            img_ids.add(ann["image_id"])

    img_ids = list(img_ids)

    return img_ids, ann_ids


def evaluate_by_supercategory(
    pred_path: str,
    JSON_ANNOT_PATH: str,
    supercats: Union[str, List],
    mode: str = "full",
):
    """
    Evaluate predictions for a specific supercategory.

    Args:
        JSON_ANNOT_PATH: Path to ground truth annotations
        pred_path: Path to predictions
        supercat_name: Name of supercategory to evaluate
    """
    assert mode in ("full", "abs", "pres"), "Invalid eval_mode"

    if mode == "full":
        JSON_ANNOT_PATH = os.path.join(
            JSON_ANNOT_PATH, "d3_full_annotations_updated.json"
        )
    elif mode == "pres":
        JSON_ANNOT_PATH = os.path.join(
            JSON_ANNOT_PATH, "d3_pres_annotations_updated.json"
        )
    else:
        JSON_ANNOT_PATH = os.path.join(
            JSON_ANNOT_PATH, "d3_abs_annotations_updated.json"
        )

    # Verify that the ground truth file exists
    if not os.path.exists(JSON_ANNOT_PATH):
        raise FileNotFoundError(
            f"Ground truth file with updated annotations not found: {JSON_ANNOT_PATH}"
        )

    # Load ground truth and predictions
    coco = COCO(JSON_ANNOT_PATH)
    coco_dt = coco.loadRes(pred_path)

    if isinstance(supercats, str):
        supercats = [supercats]

    elif isinstance(supercats, list):
        supercats = [str(name) for name in supercats]

    for name in supercats:
        # print(f"Evaluating visual supercategory: {name}")

        # Filter by supercategory
        img_ids, ann_ids = filter_by_supercategory(coco, name, mode=mode)

        print(f"Number of images with '{name}': {len(img_ids)}")
        print(f"Number of annotations with '{name}': {len(ann_ids)}")

        print(f"\n{'='*50}")
        print(f"Evaluating supercategory: {name}")
        print("=" * 50)

        coco_eval = COCOeval(coco, coco_dt, iouType="bbox")

        # Narrow evaluation to only visual category images
        coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


# Category-specific evaluation for each supercategory
def evaluate_specific_categories(
    JSON_ANNOT_PATH: str,
    pred_path: str,
    supercat_name: str,
    mode: str = "full",
):
    """
    Evaluate each category within a supercategory separately.
    """
    assert mode in ("full", "pres", "abs")
    if mode == "full":
        JSON_ANNOT_PATH = os.path.join(JSON_ANNOT_PATH, "d3_full_annotations.json")
    elif mode == "pres":
        JSON_ANNOT_PATH = os.path.join(JSON_ANNOT_PATH, "d3_pres_annotations.json")
    else:
        JSON_ANNOT_PATH = os.path.join(JSON_ANNOT_PATH, "d3_abs_annotations.json")

    coco = COCO(JSON_ANNOT_PATH)
    coco_dt = coco.loadRes(pred_path)

    # Find categories with the supercategory
    categories_to_eval = []
    for cat in coco.dataset["categories"]:
        if supercat_name in cat.get("visual_cats", []):
            categories_to_eval.append(cat)

    # print(
    #    f"Found {len(categories_to_eval)} categories with supercategory '{supercat_name}':"
    # )

    for cat in categories_to_eval:
        print(f"\nEvaluating category: {cat['name']}")
        coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
        coco_eval.params.catIds = [cat["id"]]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


# -------------------------Transformation Functions for D3-------------------------
def convert_to_xywh(x1, y1, x2, y2):
    """
    Convert top-left and bottom-right corner coordinates to [x,y,width,height] format.
    """
    if x1 > x2 or y1 > y2:
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

    width = x2 - x1
    height = y2 - y1
    return x1, y1, width, height


def transform_json_boxes(pred_path):
    with open(pred_path, "r") as f_:
        res = json.load(f_)
    for item in res:
        item["bbox"] = convert_to_xywh(*item["bbox"])
    res_path = pred_path.replace(".json", ".xywh.json")
    with open(res_path, "w") as f_w:
        json.dump(res, f_w)
    return res_path


# -------------------------Main Evaluation Function for D3-------------------------
def evaluate_with_d3(args):
    """Evaluate predictions on D3 dataset"""
    pred_path = transform_json_boxes(SAVE_PATH)  # Convert predictions to xywh format
    print(f"Transformed predictions saved to {pred_path}")

    pres_supercategories = [
        "action",
        "interaction",
        "state",
        "spatial",
        "attribute",
        "appearance",
        "object",
    ]

    abs_supercategories = [
        "negated action",
        "absent interaction",
        "negated state",
        "negated spatial",
        "absent attribute",
        "absent appearance",
    ]

    modes = ["full", "pres", "abs"]

    if args.use_supercat:
        print("Evaluating predictions by presence supercategory...")
        for supercat, mode in zip(pres_supercategories, modes):
            print("=" * 50)
            print(f"\nEvaluating presence supercategory: {supercat} in mode: {mode}")
            print("=" * 50)
            evaluate_specific_categories(
                JSON_ANNOT_PATH, pred_path, supercat, mode=mode
            )

        print("Evaluating predictions by absence supercategory...")
        for supercat, mode in zip(abs_supercategories, modes):
            print("=" * 50)
            print(f"\nEvaluating absence supercategory: {supercat} in mode: {mode}")
            print("=" * 50)
            evaluate_specific_categories(
                JSON_ANNOT_PATH, pred_path, supercat, mode=mode
            )

    elif args.use_nbox:
        # count_images_per_partition()

        for mode in modes:
            print("=" * 50)
            print(f"\nEvaluating on number of partitions in mode: {mode}")
            print("=" * 50)
            for ptt in ("no", "one", "multi"):
                # for ptt in ("no", "one", "two", "three", "four", "four_more"):
                partition_eval_on_d3(pred_path, mode=mode, nbox_partition=ptt)

    elif args.use_length:

        print("=" * 50)
        print(f"\nEvaluating on length of referring expressions in mode: {mode}")
        print("=" * 50)
        for mode in modes:
            partition_eval_on_d3(pred_path, mode=mode, lref_partition=args.use_length)

    else:
        print("Evaluating predictions on combined dataset...")
        for mode in modes:
            print("=" * 50)
            print(f"\nEvaluating in mode: {mode}")
            print("=" * 50)
            complete_evaluation(pred_path, JSON_ANNOT_PATH, mode=mode)


# -------------------------N-bbox partition -level Evaluation Function for D3-------------------------


def count_images_per_partition():
    """
    Count the number of images in each n-box partition.

    Args:
    None

    Returns:
        Dictionary with partition names and image counts
    """
    cat_obj_count = (
        d3.bbox_num_analyze()
    )  # numpy array of shape (n_categories, n_images)
    n_cat, n_img = cat_obj_count.shape

    partition_counts = {
        "zero": 0,  # No objects
        "one": 0,
        "two": 0,
        "three": 0,
        "four": 0,
        "four_more": 0,
        "multi": 0,  # More than 1 object (2+)
    }

    # Track unique images per partition
    partition_images = {
        "zero": set(),
        "one": set(),
        "two": set(),
        "three": set(),
        "four": set(),
        "four_more": set(),
        "multi": set(),
    }

    # Iterate through each category and image
    for cat_id in range(n_cat):
        for img_id in range(n_img):
            count = cat_obj_count[cat_id - 1, img_id]

            if count == 0:
                partition_images["zero"].add(img_id)
            elif count == 1:
                partition_images["one"].add(img_id)
            elif count == 2:
                partition_images["two"].add(img_id)
                partition_images["multi"].add(img_id)
            elif count == 3:
                partition_images["three"].add(img_id)
                partition_images["multi"].add(img_id)
            elif count == 4:
                partition_images["four"].add(img_id)
                partition_images["multi"].add(img_id)
            elif count > 4:
                partition_images["four_more"].add(img_id)
                partition_images["multi"].add(img_id)

    # Count unique images in each partition
    for partition, images in partition_images.items():
        partition_counts[partition] = len(images)

    print("Number of images in each partition:")
    for partition, count in partition_counts.items():
        print(f"  {partition}: {count} images")

    return partition_counts, partition_images


def nbox_partition_json(gt_path, pred_path, nbox_partition):
    with open(gt_path, "r") as f_gt:
        gts = json.load(f_gt)
    with open(pred_path, "r") as f_pred:
        preds = json.load(f_pred)

    cat_obj_count = d3.bbox_num_analyze()
    annos = gts["annotations"]
    new_annos = []
    for ann in annos:
        img_id = ann["image_id"]
        category_id = ann["category_id"]
        if nbox_partition == "one" and cat_obj_count[category_id - 1, img_id] == 1:
            new_annos.append(ann)
        if nbox_partition == "multi" and cat_obj_count[category_id - 1, img_id] > 1:
            new_annos.append(ann)
        if nbox_partition == "two" and cat_obj_count[category_id - 1, img_id] == 2:
            new_annos.append(ann)
        if nbox_partition == "three" and cat_obj_count[category_id - 1, img_id] == 3:
            new_annos.append(ann)
        if nbox_partition == "four" and cat_obj_count[category_id - 1, img_id] == 4:
            new_annos.append(ann)
        if nbox_partition == "four_more" and cat_obj_count[category_id - 1, img_id] > 4:
            new_annos.append(ann)

    gts["annotations"] = new_annos
    new_gts = gts
    new_preds = []
    for prd in preds:
        img_id = prd["image_id"]
        category_id = prd["category_id"]
        if nbox_partition == "no" and cat_obj_count[category_id - 1, img_id] == 0:
            new_preds.append(prd)
        if nbox_partition == "one" and cat_obj_count[category_id - 1, img_id] == 1:
            new_preds.append(prd)
        if nbox_partition == "multi" and cat_obj_count[category_id - 1, img_id] > 1:
            new_preds.append(prd)
        if nbox_partition == "two" and cat_obj_count[category_id - 1, img_id] == 2:
            new_preds.append(prd)
        if nbox_partition == "three" and cat_obj_count[category_id - 1, img_id] == 3:
            new_preds.append(prd)
        if nbox_partition == "four" and cat_obj_count[category_id - 1, img_id] == 4:
            new_preds.append(prd)
        if nbox_partition == "four_more" and cat_obj_count[category_id - 1, img_id] > 4:
            new_preds.append(prd)

    new_gt_path = gt_path.replace(".json", f".{nbox_partition}-instance.json")
    new_pred_path = pred_path.replace(".json", f".{nbox_partition}-instance.json")
    with open(new_gt_path, "w") as fo_gt:
        json.dump(new_gts, fo_gt)
    with open(new_pred_path, "w") as fo_pred:
        json.dump(new_preds, fo_pred)
    return new_gt_path, new_pred_path


def partition_eval_on_d3(
    pred_path, mode="pn", nbox_partition=None, lref_partition=False
):
    assert mode in ("full", "pres", "abs")
    if mode == "full":
        gt_path = os.path.join(JSON_ANNOT_PATH, "d3_full_annotations.json")
    elif mode == "pres":
        gt_path = os.path.join(JSON_ANNOT_PATH, "d3_pres_annotations.json")
    else:
        gt_path = os.path.join(JSON_ANNOT_PATH, "d3_abs_annotations.json")

    if nbox_partition:
        new_gt_path, pred_path = nbox_partition_json(gt_path, pred_path, nbox_partition)

    # Eval results
    coco = COCO(new_gt_path)
    d3_res = coco.loadRes(pred_path)
    cocoEval = COCOeval(coco, d3_res, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    aps = cocoEval.eval["precision"][:, :, :, 0, -1]
    category_ids = coco.getCatIds()
    category_names = [cat["name"] for cat in coco.loadCats(category_ids)]

    if lref_partition:
        aps_lens = defaultdict(list)
        counter_lens = defaultdict(int)
        for i in range(len(category_names)):
            ap = aps[:, :, i]
            ap_value = ap[ap > -1].mean()
            if not np.isnan(ap_value):
                len_ref = len(category_names[i].split(" "))
                aps_lens[len_ref].append(ap_value)
                counter_lens[len_ref] += 1

        ap_sum_short = sum([sum(aps_lens[i]) for i in range(0, 4)])
        ap_sum_mid = sum([sum(aps_lens[i]) for i in range(4, 7)])
        ap_sum_long = sum([sum(aps_lens[i]) for i in range(7, 10)])
        ap_sum_very_long = sum(
            [sum(aps_lens[i]) for i in range(10, max(counter_lens.keys()) + 1)]
        )
        c_sum_short = sum([counter_lens[i] for i in range(1, 4)])
        c_sum_mid = sum([counter_lens[i] for i in range(4, 7)])
        c_sum_long = sum([counter_lens[i] for i in range(7, 10)])
        c_sum_very_long = sum(
            [counter_lens[i] for i in range(10, max(counter_lens.keys()) + 1)]
        )
        map_short = ap_sum_short / c_sum_short
        map_mid = ap_sum_mid / c_sum_mid
        map_long = ap_sum_long / c_sum_long
        map_very_long = ap_sum_very_long / c_sum_very_long
        print(
            f"mAP over reference length: short - {map_short:.4f}, mid - {map_mid:.4f}, long - {map_long:.4f}, very long - {map_very_long:.4f}"
        )
        print(
            f"number of references: short - {c_sum_short}, mid - {c_sum_mid}, long - {c_sum_long}, very long - {c_sum_very_long}"
        )


def parser_args():
    parser = argparse.ArgumentParser(description="Inference on D-Cube Dataset")

    # D-cube dataset arguments
    parser.add_argument(
        "--d3_dir",
        help="Main Directory path for D-cube dataset.",
        default="../dcube/dataset",
        required=True,
    )

    parser.add_argument(
        "--json_dir",
        help="Sub directory path for JSON annotations.",
        default="d3_json",
    )

    parser.add_argument(
        "--pkl_dir",
        help="Sub directory path for annotation, groups, images and sentences pickle files.",
        default="d3_pkl",
    )

    parser.add_argument(
        "--img_dir",
        help="Sub directory path for images.",
        default="d3_images",
    )
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        # default="qwen_predictions.json",
        help="Name of the prediction file",
    )

    # Evaluation arguments

    parser.add_argument(
        "--use_supercat",
        action="store_true",  # default=False,
        help="Whether to evaluate by supercategory",
    )

    parser.add_argument(
        "--use_nbox",
        action="store_true",  # default=False,
        help="Whether to evaluate by number of bounding boxes within image",
    )

    parser.add_argument(
        "--use_length",
        action="store_true",  # default=False,
        help="Whether to evaluate by length of referring expression",
    )
    return parser.parse_args()


def main(args):
    """
    Main function to run inference with LION on the D-cube dataset.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    Returns:
        None
    """
    global IMG_ROOT, PKL_ANNO_PATH, JSON_ANNOT_PATH, CSV_ANNOT_PATH, SAVE_PATH, d3

    # Initialize paths from arguments
    IMG_ROOT = os.path.join(args.d3_dir, args.img_dir)
    PKL_ANNO_PATH = os.path.join(args.d3_dir, args.pkl_dir)
    JSON_ANNOT_PATH = os.path.join(args.d3_dir, args.json_dir)
    CSV_ANNOT_PATH = os.path.join(
        args.d3_dir, "dcube_annotated.csv"
    )  # Path to the CSV file containing visual categories annotations

    if not os.path.exists(CSV_ANNOT_PATH):
        print(
            f"Annotation file does not exist: {CSV_ANNOT_PATH}. Please check the path."
        )
        return

    # Check for predictions directory after inference
    if args.output_dir is None:
        predictions_dir = os.path.join(args.d3_dir, "predictions")
    else:
        predictions_dir = args.output_dir

    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(
            f"Predictions directory does not exist: {predictions_dir}. Please run inference first."
        )

    SAVE_PATH = os.path.join(predictions_dir, args.output_name)

    # Create visual supercategories if not already created
    run_category_mapping(JSON_ANNOT_PATH, CSV_ANNOT_PATH)

    print("Running evaluation on D-cube dataset...")
    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)
    evaluate_with_d3(args)


if __name__ == "__main__":
    args = parser_args()
    main(args)

# Usage:
# cd dcube
"""python inference/d3_evaluation.py \
    --d3_dir './dataset' \
    --output_name "{model}_predictions.json" \
"""

# to evaluate by supercategory:
"""python inference/d3_evaluation.py \
    --d3_dir './dataset' \
    --output_name "{model}_predictions.json" \
    --use_supercat
"""

# to evaluate by nbox partitions:
"""python inference/d3_evaluation.py \
    --d3_dir './dataset' \
    --output_name "{model}_predictions.json" \
    --use_nbox
"""
# to evaluate by length of referring expression:
"""python inference/d3_evaluation.py \
    --d3_dir './dataset' \
    --output_name "{model}_predictions.json" \
    --use_length
"""
