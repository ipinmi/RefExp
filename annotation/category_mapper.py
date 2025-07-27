import pandas as pd
import pickle
import argparse


def category_mapper(pkl_path, annot_path):
    """
    This function maps the sentences to their visual categories
    (both absent and present visual features).

    Args:
    pkl_path:


    Returns:
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
        required=True,
    )

    args = parser.parse_args()
    data_path = f"{args.d3_dir}/{args.pkl_dir}"
    annotation_path = f"{args.d3_dir}/dcube_annonated.csv"
    save_path = f"{args.d3_dir}/{args.pkl_dir}/visual_sentences.pkl"

    updated_sents = category_mapper(data_path, annotation_path)

    # Save sentences with visual categories as updated pickle file
    with open(save_path, "wb") as f:
        pickle.dump(updated_sents, f, protocol=pickle.HIGHEST_PROTOCOL)


# python3 annotation/category_mapper.py --d3_dir D-cube --pkl_dir d3_pkl
