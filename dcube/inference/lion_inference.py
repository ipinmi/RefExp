import os
from tqdm import tqdm
import random
import re
from PIL import Image
import argparse
import json
import torch
import gc
from typing import Dict

from models import load_model
from preprocessors.lion_preprocessors import ImageEvalProcessor
from d_cube import D3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcube.dataset.coco_evaluation import (
    complete_evaluation,
    evaluate_by_supercategory,
    run_category_mapping,
)

# -----------------------Inference with LION on D-cube Dataset-----------------------


def clear_cache():
    """Clear GPU and Python memory cache."""
    gc.collect()  # Clear Python garbage collector
    torch.cuda.empty_cache()  # Clear GPU cache


def write_predictions(preds, path):

    # Read existing data if file exists
    existing_data = []
    try:
        with open(path, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.extend(preds)

    with open(path, "w") as f:
        json.dump(existing_data, f, indent=2)


# -------------------------Image Iterator for D-cube Dataset-----------------------
def get_image_iter(d_cube):
    """
    Create an iterator for images in the D-cube dataset.

    Args:
        d_cube (D3): An instance of the D3 dataset class.
    Returns:
        img_iter (iterable): An iterator yielding tuples of (image_id, image_path).
    """
    img_ids = d_cube.get_img_ids()
    for img_id in img_ids:
        img_info = d_cube.load_imgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(IMG_ROOT, file_name)
        yield img_id, img_path


def collate_fn(batches):
    """Custom collate function for DataLoader."""

    texts = [_["text"] for _ in batches]
    hws = [_["hw"] for _ in batches]
    image_ids = [_["image_id"] for _ in batches]
    image_paths = [_["image_path"] for _ in batches]
    category_ids = [_["category_id"] for _ in batches]

    return texts, hws, image_ids, image_paths, category_ids


class DCubeDataset(torch.utils.data.Dataset):
    """Dataset class for D-Cube data."""

    def __init__(self, d_cube: D3, img_root: str):
        """
        Initialize the dataset.

        Args:
            d_cube: D3 instance for accessing D-Cube data
            prompt: Template string for formatting input
            img_root: Root directory containing images
        """
        self.img_root = img_root
        self.d_cube = d_cube
        self.data = []

        self.text_prompt_template = [
            "In the image <image>, Identify the position of {expr} in image and share its coordinates.",
            "<image> I'd like to request the coordinates of {expr} within the photo.",
            "<image> How can I locate {expr} in the image? Please provide the coordinates.",
            "<image> I am interested in knowing the coordinates of {expr} in the picture.",
            "<image> Assist me in locating the position of {expr} in the photograph and its bounding box coordinates.",
            "<image> In the image, I need to find {expr} and know its coordinates. Can you please help?",
        ]

        # Load all images and their corresponding sentences
        img_ids = d_cube.get_img_ids()

        print(f"Loading dataset with {len(img_ids)} images...")

        for img_id in tqdm(img_ids, desc="Preparing dataset"):
            img_info = d_cube.load_imgs(img_id)[0]
            image_path = os.path.join(img_root, img_info["file_name"])

            # Get all sentences for this image
            group_ids = d_cube.get_group_ids(img_ids=[img_id])
            sent_ids = d_cube.get_sent_ids(group_ids=group_ids)
            sent_list = d_cube.load_sents(sent_ids=sent_ids)
            ref_list = [sent["raw_sent"] for sent in sent_list]

            # Create one item per sentence
            for sent_id, sent in zip(sent_ids, ref_list):
                expr = sent.lower().strip()
                prompt = random.choice(self.text_prompt_template).format(
                    image=image_path, expr=expr
                )

                self.data.append(
                    {
                        "image_id": img_id,
                        "image_path": image_path,
                        "category_id": sent_id,
                        "text": prompt,
                        "hw": (img_info["height"], img_info["width"]),
                    }
                )

        print(f"Dataset prepared with {len(self.data)} sentence-image pairs")

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            "text": item["text"],
            "hw": item["hw"],
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "category_id": item["category_id"],
        }


# -------------------------Inference Function for LION-------------------------
def convert_to_xywh(x1, y1, x2, y2):
    """
    Convert top-left and bottom-right corner coordinates to [x,y,width,height] format.
    """
    # if x1 > x2 or y1 > y2:
    #    x1, x2 = min(x1, x2), max(x1, x2)
    #    y1, y2 = min(y1, y2), max(y1, y2)

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


def get_true_bbox(img_size, bbox):
    width, height = img_size
    max_edge = max(height, width)
    bbox = [v * max_edge for v in bbox]
    diff = abs(width - height) // 2
    if height < width:
        bbox[1] -= diff
        bbox[3] -= diff
    else:
        bbox[0] -= diff
        bbox[2] -= diff
    return bbox


def inference_with_lion(dataloader, model, preprocessor, save_every=50, save_path=None):
    """
    Perform inference with the LION model on the D-cube dataset.

    Args:
        dataloader (DataLoader): DataLoader for the D-cube dataset.
        model (nn.Module): The LION model.
        preprocessor (ImageEvalProcessor): The image preprocessor.
        save_every (int): How often to save predictions.
        save_path (str): Path to save predictions.

    Returns:
        predictions (list): A list of dictionaries containing predictions for each image (image_id, category_id, bbox, score, tags).
    """
    predictions = []

    for i, batch in enumerate(tqdm(dataloader, desc="Starting inference")):
        texts, hws, image_ids, image_paths, category_ids = batch

        for img_path, prompt, img_id, _, category_id in zip(
            image_paths, texts, image_ids, hws, category_ids
        ):

            # Load the image
            img = Image.open(img_path).convert("RGB")

            # Generate tags using the LION model
            tags = model.generate_tags(img)

            # Pass image through image preprocessor
            processed_img = preprocessor(img)

            output, confidence = model.generate(
                {
                    "image": processed_img.unsqueeze(0).cuda(),
                    "question": [prompt],
                    "tags": [tags],
                    "category": "region_level",
                }
            )

            # Extract the bounding box generation confidence scores if available
            if confidence is not None and len(confidence) > 2:
                bbox_confidence = confidence[3]  # Include only bbox confidence scores.
                # avg_bbox_confidence = sum(generation_confidence) / len(generation_confidence) if generation_confidence else 0.0

            # Extract the bounding box coordinates from the output
            bounding_boxes = re.search(r"\[([0-9., ]+)\]", output[0]).group(1)
            bounding_boxes = eval(bounding_boxes)  # Convert string to list/tuple
            # bounding_boxes = list(map(float, bounding_boxes))

            # bbox = get_true_bbox(img.size, bounding_boxes)

            x1, y1, x2, y2 = bounding_boxes
            # Convert to absolute coordinates
            abs_x1, abs_y1, abs_x2, abs_y2 = map(int, [x1, y1, x2, y2])

            # Save predictions to dict
            prediction = {
                "image_id": img_id,
                "image_path": img_path,
                "category_id": category_id,
                "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],  # Bounding box coordinates
                "score": (
                    float(bbox_confidence) if confidence is not None else 0.5
                ),  # Use confidence score if available, else default to 0.5
                "tags": tags,  # Tags generated by the LION model
            }

            predictions.append(prediction)

            # Save predictions to file every 'save_every' iterations and reset buffer
            if len(predictions) % save_every == 0:
                if save_path:
                    write_predictions(predictions, save_path)
                predictions = []

    # Save any remaining predictions
    if predictions and save_path:
        write_predictions(predictions, save_path)


def parser_args():
    parser = argparse.ArgumentParser(description="Inference with LION on DCube Dataset")
    # Dataloader arguments
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )

    # D-cube dataset arguments
    parser.add_argument(
        "--d3_dir",
        help="Main Directory path for D-cube dataset.",
        default="../dcube/dataset",
        required=True,
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

    parser.add_argument(
        "--json_dir",
        help="Sub directory path for JSON annotations.",
        default="d3_json",
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
        default="qwen_predictions.json",
        help="Name of the prediction file",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_only",
        action="store_true",  # default=False,
        help="Whether to only evaluate predictions after inference is done.",
    )

    parser.add_argument(
        "--use_supercat",
        action="store_true",  # default=False,
        help="Whether to evaluate by supercategory",
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
    # Initialize paths from arguments
    global IMG_ROOT, PKL_ANNO_PATH, GT_PATH, save_path, ANNOT_PATH

    IMG_ROOT = os.path.join(args.d3_dir, args.img_dir)
    PKL_ANNO_PATH = os.path.join(args.d3_dir, args.pkl_dir)
    GT_PATH = os.path.join(args.d3_dir, args.json_dir)
    ANNOT_PATH = os.path.join(
        args.d3_dir, "dcube_annotated.csv"
    )  # Path to the CSV file containing visual categories annotations

    if not os.path.exists(ANNOT_PATH):
        print(f"Annotation file does not exist: {ANNOT_PATH}. Please check the path.")
        return

    # Create predictions directory if it doesn't exist
    if args.output_dir is None:
        predictions_dir = os.path.join(args.d3_dir, "predictions")
    else:
        predictions_dir = args.output_dir

    os.makedirs(predictions_dir, exist_ok=True)
    save_path = os.path.join(predictions_dir, args.output_name)

    # Create visual supercategories if not already created
    run_category_mapping(GT_PATH, ANNOT_PATH)

    if args.eval_only:
        print("Running evaluation only...")

        evaluate_with_d3(args)

        return

    # Load the LION model and preprocessor
    lion_model = load_model("lion_t5", "flant5xl", is_eval=True, device="cuda:0")

    # Initialize the preprocessor for image evaluation
    lion_preprocessor = ImageEvalProcessor()

    # Initialize D-Cube instance
    print("Loading D-Cube dataset...")
    D_3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    # Create the dataset
    dataset = DCubeDataset(d_cube=D_3, img_root=IMG_ROOT)

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    try:
        # Clear cache before inference
        clear_cache()

        # Run inference with LION model
        inference_with_lion(
            dataloader,
            lion_model,
            lion_preprocessor,
            save_every=100,
            save_path=save_path,
        )

        print(f"Inference completed. Predictions saved to {save_path}")

    except RuntimeError as e:
        print(f"GPU out of memory: {e}")
        clear_cache()

    evaluate_with_d3(args)  # Evaluate predictions after inference


def evaluate_with_d3(args):
    """Evaluate predictions on D3 dataset"""
    pred_path = transform_json_boxes(save_path)  # Convert predictions to xywh format
    print(f"Transformed predictions saved to {pred_path}")

    if args.use_supercat:
        print("Evaluating predictions by supercategory...")
        evaluate_by_supercategory(pred_path, GT_PATH, mode="full")  # Full evaluation
        evaluate_by_supercategory(
            pred_path, GT_PATH, mode="pres"
        )  # Presence evaluation
        evaluate_by_supercategory(pred_path, GT_PATH, mode="abs")  # Absence evaluation
    else:
        print("Evaluating predictions on combined dataset...")
        complete_evaluation(pred_path, GT_PATH, mode="full")  # Full evaluation
        complete_evaluation(pred_path, GT_PATH, mode="pres")  # Presence evaluation
        complete_evaluation(pred_path, GT_PATH, mode="abs")  # Absence evaluation


if __name__ == "__main__":
    args = parser_args()
    main(args)


# Usage:
# cd JiuTian-LION
# Only inference:
"""
python lion_inference.py \
    --batch-size 8 \
    --d3_dir "../dcube/dataset" \
    --output_name "lion_predictions.json" \
"""

# With only evaluation:
"""python lion_inference.py \
    --batch-size 8 \
    --d3_dir "../dcube/dataset" \
    --output_name "lion_predictions.json" \
    --eval_only  \
    --use_supercat  \
"""
