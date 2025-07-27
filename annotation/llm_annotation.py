import ollama
import pandas as pd


def batch_annotate_csv(csv_path, modelName, batch_size=5):
    client = ollama.Client()

    # Annotated data
    annotated_data = []

    annotation_df = pd.read_csv(csv_path)

    for i in range(0, len(annotation_df), batch_size):
        batch = annotation_df["cleaned_expression"].iloc[i : i + batch_size]

        # Create base batch prompt
        prompt = "Analyze these texts and find broad visual categories for how the referred objects are referred to and give a brief justification (1-2 sentences):\n"

        prompt_examples = "Here are some examples: The woman's hat | Spatial | Indicates that the hat is on the woman's head.\n"
        prompt_examples += (
            "The red apple | Attribute | Indicates the color of the apple.\n"
        )
        prompt_examples += (
            "The dancing man | Action | Indicates that the man is performing a dance.\n"
        )
        prompt += prompt_examples

        # Add batch items to the prompt
        prompt += "Here are the texts:\n"
        for j, text in enumerate(batch):
            prompt += f"{i+j+1}: {text}\n"

        prompt += "Respond strictly in this format:\n X: [text] | [category] | [justification] with no other comments.\n"

        response = client.chat(
            model=modelName, messages=[{"role": "user", "content": prompt}]
        )

        # Process the response
        response_content = response["message"]["content"]
        responses = response_content.strip().split("\n")
        for line in responses:
            parts = line.split("|")
            if len(parts) == 3:
                annotated_data.append(
                    {
                        "text": parts[0].strip(),
                        "category": parts[1].strip(),
                        "justification": parts[2].strip(),
                    }
                )

        # print(response_content)
        print(
            f"Processed batch {i//batch_size + 1} of {len(annotation_df)//batch_size + 1}"
        )

    # Create a DataFrame from the annotated data
    annotated_df = pd.DataFrame(annotated_data)
    print(f"Total annotations collected: {len(annotated_df)}")

    # Save the annotated DataFrame to a new CSV file
    annotated_df.to_csv("labelled_data/annotated_data.csv", index=False)


if __name__ == "__main__":

    csv_path = "category_annotations_cleaned.csv"
    model_name = "gemma3"
    batch_annotate_csv(csv_path, modelName=model_name, batch_size=15)
