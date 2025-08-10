import os
from tqdm import tqdm
import random
import re
from PIL import Image
import argparse
import json
import torch


from models import load_model
from preprocessors.lion_preprocessors import ImageEvalProcessor
from d_cube import D3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcube.dataset.coco_evaluation import complete_evaluation, evaluate_by_supercategory

# -----------------------Inference with LION on D-cube Dataset-----------------------

# Load the LION 4B and text LLM models
lion_model = load_model("lion_t5", "flant5xl", is_eval=True, device="cuda:0")

# Initialize the preprocessor for image evaluation
lion_preprocessor = ImageEvalProcessor()

# Define the text prompt template for LION model
text_prompt_template = [
    "In the image <image>, Identify the position of {expr} in image and share its coordinates.",
    "<image> I'd like to request the coordinates of {expr} within the photo.",
    "<image> How can I locate {expr} in the image? Please provide the coordinates.",
    "<image> I am interested in knowing the coordinates of {expr} in the picture.",
    "<image> Assist me in locating the position of {expr} in the photograph and its bounding box coordinates.",
    "<image> In the image, I need to find {expr} and know its coordinates. Can you please help?",
]


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


def inference_with_lion(img_iter, d_cube, save_path):
    """
    Perform inference with the LION model on the D-cube dataset.

    Args:
        img_iter (iterable): An iterator yielding tuples of (image_id, image_path).
        d_cube (D3): An instance of the D3 dataset class.
        save_path (str): Path to save the predictions.

    Returns:
        predictions (list): A list of dictionaries containing predictions for each image (image_id, category_id, bbox, score, tags).
    """
    predictions = []
    save_every = 50

    for img_id, image_path in tqdm(
        img_iter, desc="Processing referring expressions", total=(4 * 10578)
    ):

        # Load the image
        img = Image.open(image_path).convert("RGB")

        # Generate tags using the LION model
        tags = lion_model.generate_tags(img)

        # Pass image through image preprocessor
        processed_img = lion_preprocessor(img)

        # Generate references expressions for the image in the D cube dataset
        group_ids = d_cube.get_group_ids(
            img_ids=[img_id]
        )  # each image is evaluated with the categories in its group (usually 4)
        sent_ids = d_cube.get_sent_ids(
            group_ids=group_ids
        )  # get sentence ids for expressions in the group
        sent_list = d_cube.load_sents(
            sent_ids=sent_ids
        )  # load the sentences from sent_ids
        ref_list = [
            sent["raw_sent"] for sent in sent_list
        ]  # referring expressions for the image

        # Select a random prompt template for each expression
        for sent_id, expr in zip(sent_ids, ref_list):
            # Format the text prompt with the image path and expression
            text_prompt = random.choice(text_prompt_template).format(
                image=image_path, expr=expr
            )
            text_prompt = text_prompt.lower()  # Convert to lowercase for consistency
            text_prompt = text_prompt.strip()  # Remove leading/trailing whitespace

            output, confidence = lion_model.generate(
                {
                    "image": processed_img.unsqueeze(0).cuda(),
                    "question": [text_prompt],
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
            bounding_boxes = eval(bounding_boxes)  # Convert string to list of floats

            x1, y1, x2, y2 = [
                bounding_boxes[0] * img.width,
                bounding_boxes[1] * img.height,
                bounding_boxes[2] * img.width,
                bounding_boxes[3] * img.height,
            ]

            # Save predictions to dict
            prediction = {
                "image_id": img_id,
                "image_path": image_path,
                "category_id": sent_id,
                "bbox": [x1, y1, x2, y2],  # Bounding box coordinates
                "score": (
                    float(bbox_confidence) if confidence is not None else "n/a"
                ),  # Use confidence score if available, else default to 1.0
                "tags": tags,  # Tags generated by the LION model
            }

            predictions.append(prediction)

            # Save predictions to file every 'save_every' iterations and reset buffer
            if len(predictions) % save_every == 0:
                write_predictions(predictions, save_path)
                predictions = []

    # Save any remaining predictions
    if predictions:
        write_predictions(predictions, save_path)


def parser_args():
    parser = argparse.ArgumentParser(description="Inference with LION on DCube Dataset")

    parser.add_argument(
        "--d3_dir",
        help="Main Directory path for D-cube dataset.",
        default="dcube",
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
        default="predictions",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="qwen_predictions.json",
        help="Name of the prediction file",
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
    global IMG_ROOT, PKL_ANNO_PATH, GT_PATH, save_path

    IMG_ROOT = os.path.join(args.d3_dir, args.img_dir)
    PKL_ANNO_PATH = os.path.join(args.d3_dir, args.pkl_dir)
    GT_PATH = os.path.join(args.d3_dir, args.json_dir)

    # Create predictions directory if it doesn't exist
    if args.output_dir is None:
        predictions_dir = os.path.join(args.d3_dir, "predictions")
    else:
        predictions_dir = args.output_dir

    os.makedirs(predictions_dir, exist_ok=True)
    save_path = os.path.join(predictions_dir, args.output_name)

    # Initialize D-Cube instance
    print("Loading D-Cube dataset...")
    D_3 = D3(IMG_ROOT, PKL_ANNO_PATH)

    # Get image iterator
    img_iter = get_image_iter(D_3)

    try:
        # Clear cache before inference
        clear_cache()

        # Run inference with LION model
        inference_with_lion(img_iter, D_3, save_path)

        # Convert JSONL to JSON array format
        convert_jsonl_to_array(save_path, save_path)

    except RuntimeError as e:
        print(f"GPU out of memory: {e}")
        clear_cache()


def eval(args):
    # Evaluate predictions on D3 dataset
    if args.use_supercat:
        print("Evaluating predictions by supercategory...")
        evaluate_by_supercategory(save_path, GT_PATH, mode="full")  # Full evaluation
        evaluate_by_supercategory(
            save_path, GT_PATH, mode="pres"
        )  # Presence evaluation
        evaluate_by_supercategory(save_path, GT_PATH, mode="abs")  # Absence evaluation
    else:
        print("Evaluating predictions on combined dataset...")
        complete_evaluation(save_path, GT_PATH, mode="full")  # Full evaluation
        complete_evaluation(save_path, GT_PATH, mode="pres")  # Presence evaluation
        complete_evaluation(save_path, GT_PATH, mode="abs")  # Absence evaluation


if __name__ == "__main__":
    args = parser_args()
    main(args)

    if args.eval:
        eval(args)


# Usage:
# cd JiuTian-LION
"""
python lion_inference.py \
    --d3_dir "../dcube/dataset" \
    --pkl_dir "d3_pkl" \
    --img_dir "d3_images" \
    --json_dir "d3_json" \
    --output_dir "predictions" \
    --output_name "lion_predictions.json" \
    --eval False 
"""
