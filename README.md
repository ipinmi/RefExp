# Look Again: Exploring the Cracks in Visual Grounding Models

Project on Referring Expression Comprehension for Language &amp; Vision seminar (SS25)

## Contents

- [D-Cube dataset](#d-cube-dataset)
- [File Structure](#file-structure-for-inference)
- [Setup of MLLMs](#setup-of-mllms)

## D-Cube dataset

The Description Detection Dataset ($D^3$) introduces a new task, Described Object
Detection (DOD), as a superset of two vision-language tasks: Open-Vocabulary Object Detection(OVD) and Referring Expression Comprehension (REC). DOD addresses the limitations of both: unlike OVD, it handles detailed descriptions beyond simple categories, and unlike REC, it can verify the existence of a described object. Full details can be found in the [$D^3$ paper](https://arxiv.org/abs/2307.12813) and [repository](https://github.com/shikras/d-cube).

### Preliminary Setup

1. Clone our project repository

   ```bash
   git clone https://github.com/ipinmi/RefExp.git
   cd RefExp
   ```

2. Download the D-Cube dataset from the [official repository](https://github.com/shikras/d-cube?tab=readme-ov-file#download) and move into the **existing** folder `dcube/dataset`. The dataset is structured as follows:

   ```tree
   d-cube/
      ├── dataset/
         ├── d3_images/          # Directory containing images
         ├── d3_json/            # Directory containing JSON files with annotations for each setting
         ├── d3_pkl/             # Directory containing pickle files with annotations, groups, images, and sentences
   ```

## File Structure For Inference

Organization of the inference scripts and models in the repository is as follows:

```tree
RefExp/
      d-cube/
      |   |── dataset/
      |   |  ├── d3_images/
      |   |  ├── d3_json/
      |   |  ├── d3_pkl/
      |   |  ├── predictions/

      |   ├── inference/
      |   |  ├── lion_inference.py
      |   |  ├── qwen_inference.py
      |   |  ├── fix_json.py
      |   |  ├── d3_evaluation.py

      Qwen2.5VL/
      |   |── qwen_inference.py
      |   |── fix_json.py

      JiuTian-LION/
      |   |── lion_inference.py

```

## Setup of MLLMs

We evaluated the following MLLMs on the D-Cube dataset:

- [LION](https://github.com/JiuTian-VL/JiuTian-LION)
- [QWEN2.5 VL 7B](https://github.com/QwenLM/Qwen2.5-VL)

### LION Setup

1. Clone the repository in the same directory as the D-Cube dataset:

   ```bash
   git clone https://github.com/JiuTian-VL/JiuTian-LION.git
   cd JiuTian-LION
   ```

2. Create a virtual environment and install the required dependencies:

   ```bash
    conda create -n LION python=3.12
    conda activate LION
    conda install pip
    pip install -r requirements.txt
   ```

3. Download the checkpoints and pre-trained model weights following the instructions in the [LION repository](https://github.com/JiuTian-VL/JiuTian-LION).

4. Install the D-Cube dataset:

   ```bash
   pip install ddd-dataset
   ```

5. Move inference script `lion_inference.py` to the root directory of the cloned repository.

   ```bash
   cp ../dcube/inference/lion_inference.py .
   ```

6. **Inference** can be run using the following command:

   ```bash
   python lion_inference.py \
      --batch-size {BATCH_SIZE} \
      --d3_dir {D_CUBE_DIR} \
      --output_name "lion_predictions.json" \
   ```

7. **Evaluation** can be run using the following command:

   ```bash
   cd dcube
   python inference/d3_evaluation.py \
      --d3_dir './{D_CUBE_DIR}' \
      --output_name "lion_predictions.json" \
      --use_supercat
   ```

   > NOTE: The `--use_supercat` flag is optional and can be used to evaluate by supercategory.

### QWEN2.5 VL Setup

Model Card for QWEN2.5 VL series can be found on [HuggingFace](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5). We use the [AWQ quantized 7B version](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) of the model.

1. Create a virtual environment:

   ```bash
   conda create -n QWEN2.5 python=3.12
   conda activate QWEN2.5
   ```

2. Install torch and torchvision based on your CUDA version. We use CUDA 12.6:

   ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

3. Build the transformers library from source:

   ```bash
   pip install git+https://github.com/huggingface/transformers accelerate
   ```

4. Install the D-Cube dataset:

   ```bash
   pip install ddd-dataset
   ```

5. **Optional**: Install the AWQ quantization library:

   ```bash
   pip install autoawq
   ```

   > AutoAWQ downgrades Transformers to version 4.47.1. If you want to do inference with AutoAWQ, you may need to reinstall the Transformers’ version after installing AutoAWQ.

6. **Optional**: Enable flash_attention_2 for better acceleration and memory saving.

   > NOTE: The installation takes up to 1-3 hours.

   ```bash
   pip install flash-attn --no-build-isolation
   ```

7. Move inference script `qwen_inference.py` and json processing script `fix_json.py` to the root directory of the cloned repository.

   ```bash
   cp ../dcube/inference/qwen_inference.py .
   cp ../dcube/inference/fix_json.py .
   ```

8. **Inference** can be run using the following command:

   ```bash
   python qwen_inference.py \
   --checkpoint {HF_CHECKPOINT} \
   --use-flash-attention True \
   --batch-size {BATCH_SIZE} \
   --d3_dir {D_CUBE_DIR} \
   --output_name "qwen2.5_predictions.json"
   ```

9. **Evaluation** can be run using the following command:

   ```bash
   cd dcube
   python inference/d3_evaluation.py \
      --d3_dir './{D_CUBE_DIR}' \
      --output_name "qwen2.5_predictions.json" \
   ```
