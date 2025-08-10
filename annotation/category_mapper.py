import pandas as pd
import pickle
import argparse
import json


def category_mapper(pkl_path, annot_path):
    """
    This function maps the sentences to their visual categories
    (both absent and present visual features).

    Args:
    pkl_path: Path to the directory containing the sentences pickle file.
    annot_path: Path to the annotated dataset CSV file.


    Returns:
    sentences: A dictionary where each key is a sentence ID and the value is a dictionary
               containing the existing sentence annotations and its associated visual categories.
    """
    with open(f"{pkl_path}/sentences.pkl", "rb") as f:
        sentences = pickle.load(f)

    # Load annotated dataset
    annotated_df = pd.read_csv(annot_path, sep=";")
    annotated_df.columns = annotated_df.columns.str.strip()

    # Add the visual category to existing dataset
    category_cols = [col for col in annotated_df.columns if col.startswith("category_")]

    for idx, row in annotated_df.iterrows():
        visual_cats = [row[col] for col in category_cols if pd.notna(row[col])]

        sent_id = row["id"]
        if sent_id in sentences:
            sentences[sent_id]["visual_id"] = visual_cats

    return sentences


def cocoapi_supercategory(json_path, annot_path, eval_mode="full"):
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
    save_path = f"{json_path}/d3_{eval_mode}_annotations_updated.json"
    with open(save_path, "w") as f:
        json.dump(gt_annotations, f, indent=2)

    print(f"Updated annotations for {eval_mode} saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description Detection Dataset")

    parser.add_argument(
        "--d3_dir",
        help="Main Directory path for D-cube dataset.",
        default="d3_dir",
        required=True,
    )

    parser.add_argument(
        "--pkl_dir",
        help="Sub directory path for annotation, groups, images and sentences pickle files.",
        default="d3_pkl",
    )

    parser.add_argument(
        "--json_dir",
        help="Sub directory path for annotation, groups, images and sentences pickle files.",
        default="d3_json",
    )

    args = parser.parse_args()
    pkl_path = f"{args.d3_dir}/{args.pkl_dir}"
    json_path = f"{args.d3_dir}/{args.json_dir}"
    annotation_path = f"{args.d3_dir}/dcube_annonated.csv"

    """
    save_path = f"{args.d3_dir}/{args.pkl_dir}/visual_sentences.pkl"
    updated_sents = category_mapper(pkl_path, annotation_path)

    # Save sentences with visual categories as updated pickle file
    with open(save_path, "wb") as f:
        pickle.dump(updated_sents, f, protocol=pickle.HIGHEST_PROTOCOL)
    """
    
    cocoapi_supercategory(json_path, annotation_path, eval_mode="full")
    cocoapi_supercategory(json_path, annotation_path, eval_mode="pres")
    cocoapi_supercategory(json_path, annotation_path, eval_mode="abs")

# python3 annotation/category_mapper.py --d3_dir D-cube --pkl_dir d3_pkl --json_dir d3_json
