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
            bounding_boxes = eval(bounding_boxes)  # Convert string to tuple of floats

            x1, y1, x2, y2 = [
                bounding_boxes[0] * img.width,
                bounding_boxes[1] * img.height,
                bounding_boxes[2] * img.width,
                bounding_boxes[3] * img.height,
            ]
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
        default="dcube/dataset",
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

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="flant5xl",
        help="Name of the model to use for inference. Options: flant5xxl for lion 12B, flant5xl for lion 4B",
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

    # Create predictions directory if it doesn't exist
    if args.output_dir is None:
        predictions_dir = os.path.join(args.d3_dir, "predictions")
    else:
        predictions_dir = args.output_dir

    os.makedirs(predictions_dir, exist_ok=True)
    save_path = os.path.join(predictions_dir, args.output_name)

    # Load the LION model and preprocessor
    lion_model = load_model("lion_t5", args.model_type, is_eval=True, device="cuda:0")

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


if __name__ == "__main__":
    args = parser_args()
    main(args)


# Usage:
# cd JiuTian-LION
"""
python lion_inference.py \
    --batch-size 8 \
    --d3_dir "../dcube/dataset" \
    --output_name "lion_predictions.json" \
"""
