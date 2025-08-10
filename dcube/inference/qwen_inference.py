import torch
import json
import gc
import os
from typing import List, Dict
from tqdm import tqdm
import random
import ast
import argparse

from d_cube import D3
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcube.dataset.coco_evaluation import (
    complete_evaluation,
    evaluate_by_supercategory,
    evaluate_specific_categories,
)


def parse_json(json_bbox):
    # Parsing out the markdown fencing
    lines = json_bbox.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_bbox = "\n".join(lines[i + 1 :])  # Remove everything before "```json"
            json_bbox = json_bbox.split("```")[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_bbox


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


def collate_fn(batches):
    """Custom collate function for DataLoader."""

    texts = [_["text"] for _ in batches]
    hws = [_["hw"] for _ in batches]
    image_ids = [_["image_id"] for _ in batches]
    image_paths = [_["image_path"] for _ in batches]

    return texts, hws, image_ids, image_paths


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

        """self.text_prompt_template = [
            "Identify the position of {expr} in image and share the coordinates in JSON format. Answer with all zero coordinates in JSON format if it does not exist.",
            "I'd like to request the coordinates of {expr} within the photo in JSON format. Answer with all zero coordinates in JSON format if it does not exist.",
            "How can I locate {expr} in the image? Please provide the coordinates in JSON format. Answer with all zero coordinates in JSON format if it does not exist.",
            "I am interested in knowing the coordinates of {expr} in the picture in JSON format. Answer with all zero coordinates in JSON format if it does not exist.",
            "Assist me in locating the position of {expr} in the photograph and the bounding box coordinates in JSON format. Answer with all zero coordinates in JSON format if it does not exist.",
            "In the image, I need to find {expr} and know the coordinates in JSON format. Can you please help? Answer with all zero coordinates in JSON format if it does not exist.",
        ]"""

        self.text_prompt_template = """Task: Locate all bounding box coordinates for the referring expression in the provided image.
            Input:
            - Image: [image]
            - Referring expression: {expr}

            Output format:
            Return an array containing all matching objects in a valid JSON format.

            Rules:
            1. If the referring expression matches one or more objects, return all their bounding boxes.
            2. If the referring expression does not exist in the image, return the array as [{{"bbox_2d": [0, 0, 0, 0], "label": "{expr}"}}]
            3. Each detected instance gets its own object in the array
            4. The "label" field must contain the exact referring expression for all detections"""

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
            for sent in ref_list:
                expr = sent.lower().strip()
                prompt = self.text_prompt_template.format(expr=expr)

                self.data.append(
                    {
                        "image_id": img_id,
                        "image_path": image_path,
                        "text": prompt,
                        "hw": (img_info["height"], img_info["width"]),
                    }
                )

        print(f"Dataset prepared with {len(self.data)} sentence-image pairs")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            "text": item["text"],
            "hw": item["hw"],
            "image_id": item["image_id"] + 1,
            "image_path": item["image_path"],
        }


def qwen_inference(
    img_path,
    prompt,
    model,
    processor,
    system_prompt,
    max_tokens,
):
    """
    Perform inference using the QWEN model.

    Args:
        img_path (str): Path to the input image.
        prompt (str): Text prompt for the model.
        model (Qwen2_5_VLForConditionalGeneration): The QWEN model instance.
        processor (AutoProcessor): The processor for handling inputs and outputs.
        system_prompt (str): System prompt for the model.
        max_tokens (int): Maximum number of additional tokens to generate.

    Returns:
        response: The generated output text.
        input_height (int): Height of the input image grid.
        input_width (int): Width of the input image grid.
    """

    image = Image.open(img_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, {"image": img_path}],
        },
    ]

    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Restore the input height and width based on the image grid dimensions
    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    return output_text[0], input_height, input_width


def convert_to_xywh(bbox_xyxy):
    """
    Convert top-left and bottom-right corner coordinates to [x, y, width, height] format.
    """
    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    return [x1, y1, width, height]


def run_inference(
    dataloader: torch.utils.data.DataLoader,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    system_prompt: str = "You are a helpful assistant",
    max_tokens: int = 1024,
    save_every: int = 50,
    save_path: str = None,
) -> List[Dict]:

    outputs = []

    for i, batch in enumerate(tqdm(dataloader, desc="Starting inference")):
        # Starting from i = 5378 (where it broke)
        if i < 5378:
            continue

        texts, hws, image_ids, image_paths = batch

        for img_path, prompt, img_id, (img_height, img_width) in zip(
            image_paths, texts, image_ids, hws
        ):

            response, input_height, input_width = qwen_inference(
                img_path=img_path,
                prompt=prompt,
                model=model,
                processor=processor,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )

            bbox = parse_json(response)  # Parse the JSON output

            try:
                json_bbox = ast.literal_eval(bbox)
            except Exception as e:
                end_idx = bbox.rfind('"}') + len('"}')
                truncated_text = bbox[:end_idx] + "]"
                json_bbox = ast.literal_eval(truncated_text)

            # print(f"Image ID: {img_id}, Response: {json_bbox}")

            # If response contains multiple bounding boxes, create individual outputs
            for bbox in json_bbox:
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = int(bbox["bbox_2d"][1] / input_height * img_height)
                abs_x1 = int(bbox["bbox_2d"][0] / input_width * img_width)
                abs_y2 = int(bbox["bbox_2d"][3] / input_height * img_height)
                abs_x2 = int(bbox["bbox_2d"][2] / input_width * img_width)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                outputs.append(
                    {
                        "image_id": img_id,
                        "bbox": [
                            abs_x1,
                            abs_y1,
                            abs_x2,
                            abs_y2,
                        ],  # convert_to_xywh([abs_x1, abs_y1, abs_x2, abs_y2]),
                        "score": 1.0,  # TODO: Add score logic
                    }
                )

            # Save predictions to file every 'save_every' iterations and reset buffer
            if len(outputs) % save_every == 0:
                if save_path:
                    write_predictions(outputs, save_path)
                outputs = []

    # Save any remaining predictions
    if outputs and save_path:
        write_predictions(outputs, save_path)

    return outputs


def parser_args():
    parser = argparse.ArgumentParser(
        description="Single GPU inference with QWEN2.5-VL model on D-Cube dataset"
    )

    # QWEN2.5-VL model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to QWEN2.5-VL model checkpoint",
    )
    # use flash attention for better performance
    parser.add_argument(
        "--use-flash-attention",
        type=bool,
        default=True,
        help="Use flash attention for better performance",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )

    # D-Cube dataset paths
    parser.add_argument(
        "--d3_dir",
        type=str,
        default="dcube",
        help="Main directory path for D-cube dataset",
    )
    parser.add_argument(
        "--pkl_dir", type=str, default="d3_pkl", help="Sub directory for pickle files"
    )
    parser.add_argument(
        "--img_dir", type=str, default="d3_images", help="Sub directory for images"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="d3_json",
        help="Sub directory for JSON annotations",
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

    # Evaluation arguments
    parser.add_argument(
        "--eval",
        type=bool,
        default=False,
        help="Whether to evaluate predictions after inference",
    )

    parser.add_argument(
        "--use_supercat",
        type=bool,
        default=False,
        help="Whether to evaluate by supercategory",
    )

    return parser.parse_args()


def main(args):

    # Load the QWEN2.5-VL model
    print(f"Loading model from {args.checkpoint}...")

    clear_cache()  # Clear cache before loading model

    if args.use_flash_attention:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint,
            torch_dtype="auto",
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(args.checkpoint)

    # Initialize the D3 dataset paths
    IMG_ROOT = os.path.join(args.d3_dir, args.img_dir)
    PKL_ANNO_PATH = os.path.join(args.d3_dir, args.pkl_dir)
    GT_PATH = os.path.join(args.d3_dir, args.json_dir)

    # Create predictions directory
    if args.output_dir is None:
        predictions_dir = os.path.join(args.d3_dir, "predictions")
    else:
        predictions_dir = args.output_dir

    os.makedirs(predictions_dir, exist_ok=True)
    save_path = os.path.join(predictions_dir, args.output_name)

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

    # Run inference
    try:
        outputs = run_inference(
            dataloader=dataloader,
            model=model,
            processor=processor,
            system_prompt="You are a helpful assistant",
            max_tokens=1024,
            save_every=100,
            save_path=save_path,
        )
        print(f"Inference completed. Predictions saved to {save_path}")
    except RuntimeError as e:
        print(f"GPU out of memory: {e}")
        clear_cache()


def eval(args):
    GT_PATH = os.path.join(args.d3_dir, args.json_dir)

    # Create predictions directory
    if args.output_dir is None:
        predictions_dir = os.path.join(args.d3_dir, "predictions")
    else:
        predictions_dir = args.output_dir

    save_path = os.path.join(predictions_dir, args.output_name)
    if not os.path.exists(save_path):
        print(f"Prediction file {save_path} does not exist. Skipping evaluation.")
        return

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
        evaluate_specific_categories(save_path, GT_PATH, mode="full")  # Full evaluation
        evaluate_specific_categories(
            save_path, GT_PATH, mode="pres"
        )  # Presence evaluation
        evaluate_specific_categories(
            save_path, GT_PATH, mode="abs"
        )  # Absence evaluation


if __name__ == "__main__":

    args = parser_args()
    main(args)

    if args.eval:
        eval(args)


# Usage:
# cd Qwen2.5VL
"""
python qwen_inference.py \
    --checkpoint "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" \
    --use-flash-attention True \
    --d3_dir "../dcube/dataset" \
    --output_dir "predictions" \
    --output_name "qwen2.5_predictions.json" \
    --eval True \
    --use_supercat True
"""
