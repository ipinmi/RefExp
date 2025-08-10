import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def filter_by_supercategory(coco, supercat_name):
    """
    Filter annotations based on supercategory.

    Args:
        coco: COCO object
        supercat_name: Name of the supercategory to filter by. (Spatial, object, attribute, action, interaction, appearance, and state.)

    Returns:
        img_ids: List of image IDs containing annotations from this supercategory
        ann_ids: List of annotation IDs belonging to this supercategory
    """

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


def evaluate_by_supercategory(gt_path, pred_path, supercat_name):
    """
    Evaluate predictions for a specific supercategory.

    Args:
        gt_path: Path to ground truth annotations
        pred_path: Path to predictions
        supercat_name: Name of supercategory to evaluate
    """
    # Load ground truth and predictions
    coco = COCO(gt_path)
    coco_dt = coco.loadRes(pred_path)

    # Filter by supercategory
    img_ids, ann_ids = filter_by_supercategory(coco, supercat_name)

    # print(f"Number of images with '{supercat_name}': {len(img_ids)}")
    # print(f"Number of annotations with '{supercat_name}': {len(ann_ids)}")

    if len(img_ids) == 0:
        print(
            f"No images found for supercategory '{supercat_name}'. Enter a valid supercategory name."
        )
        return

    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")

    # Narrow evaluation to only visual category images
    coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def full_evaluation(pred_path, gt_path, mode="full"):
    """
    Evaluate predictions on the D3 dataset.
    Args:
        pred_path (str): Path to the predictions JSON file.
        gt_path (str): Path to the ground truth annotations directory.
        mode (str): Evaluation mode - "full", "pres", or "abs".

    Returns:
        None: Prints evaluation results to the console.
    """
    assert mode in ("full", "pres", "abs")
    if mode == "full":
        gt_path = os.path.join(gt_path, "d3_full_annotations.json")
    elif mode == "pres":
        gt_path = os.path.join(gt_path, "d3_pres_annotations.json")
    else:
        gt_path = os.path.join(gt_path, "d3_abs_annotations.json")

    # Load ground truth and predictions using COCO API
    coco = COCO(gt_path)
    d3_res = coco.loadRes(pred_path)
    cocoEval = COCOeval(coco, d3_res, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


# Category-specific evaluation for each supercategory
def evaluate_specific_categories(gt_path, pred_path, supercat_name):
    """
    Evaluate each category within a supercategory separately.
    """
    coco = COCO(gt_path)
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
