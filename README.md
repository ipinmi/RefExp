# Look Again: Exploring the Cracks in Visual Grounding Models

Project on Referring Expression Comprehension for Language &amp; Vision seminar (SS25)

## Contents

- [D-Cube dataset](#d-cube-dataset)
- [Setup of MLLMs](#setup-of-mllms)
- [Evaluation Results](#evaluation-results)

## D-Cube dataset

The Description Detection Dataset ($D^3$) introduces a new task, Described Object
Detection (DOD), as a superset of two vision-language tasks: Open-Vocabulary Object Detection(OVD) and Referring Expression Comprehension (REC). DOD addresses the limitations of both: unlike OVD, it handles detailed descriptions beyond simple categories, and unlike REC, it can verify the existence of a described object. Full details can be found in the [$D^3$ paper](https://arxiv.org/abs/2307.12813) and repository [here](https://github.com/shikras/d-cube).

<!-- Add example from dataset -->

> Download the D-Cube dataset from the [official repository](https://github.com/shikras/d-cube?tab=readme-ov-file#download) and move into the folder `dcube/dataset`. The dataset is structured as follows:

```d-cube/
├── d3_images/          # Directory containing images
├── d3_json/            # Directory containing JSON files with annotations for each setting
├── d3_pkl/             # Directory containing pickle files with annotations, groups, images, and sentences
```

## Setup of MLLMs

<!-- File structure to be added -->

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
   mv dcube/inference/lion_inference.py .
   ```

6. Inference can be run using the following command:

   ```bash
   python lion_inference.py \
       --d3_dir {D_CUBE_DIR} \
       --pkl_dir "d3_pkl" \
       --img_dir "d3_images" \
       --json_dir "d3_json" \
       --output_name "lion_predictions.json"
   ```

### QWEN2.5 VL Setup

Model Card for QWEN2.5 VL series can be found [here](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5). We use the [AWQ quantized 7B version](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) of the model.

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

7. Move inference script `qwen_inference.py` to the root directory of the cloned repository.

   ```bash
   mv dcube/inference/qwen_inference.py .
   ```

8. Inference can be run using the following command:

   ```bash
   python qwen_inference.py \
   --checkpoint {HF_CHECKPOINT} \
   --use-flash-attention True \
   --d3_dir {D_CUBE_DIR} \
   --output_name "qwen2.5_predictions.json"
   ```

## Evaluation Results
