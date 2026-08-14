#!/bin/bash
set -e

echo "Starting SAM3 Compilation Pipeline..."

# 1. Setup Directories
echo "Setting up directories..."
mkdir -p "$(pwd)/model_onnx"

mkdir -p "$(pwd)/model_repository/vision_encoder/1"
mkdir -p "$(pwd)/model_repository/text_encoder/1"
mkdir -p "$(pwd)/model_repository/mask_decoder/1"

# 2. Download ONNX Models
echo "Downloading SAM3 ONNX models..."
cd "$(pwd)/model_onnx"
wget -nc https://github.com/jamjamjon/assets/releases/download/sam3/vision-encoder.onnx
wget -nc https://github.com/jamjamjon/assets/releases/download/sam3/text-encoder.onnx
wget -nc https://github.com/jamjamjon/assets/releases/download/sam3/decoder.onnx
cd -

echo "Downloading Tokenizer assets..."
wget -nc -P model_repository/sam3_model/1/tokenizer_assets https://github.com/jamjamjon/assets/releases/download/sam3/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json}


# 3. Build TensorRT Engines
echo "Compiling TensorRT Engines (This will take a while)..."

# using rtx 5060 ti
docker run --rm --gpus all \
  -v $(pwd)/model_onnx:/models \
  -v $(pwd)/model_repository:/repo \
  nvcr.io/nvidia/tensorrt:26.05-py3 bash -c "
  
  echo 'Compiling Vision Encoder...'
  trtexec --onnx=/models/vision-encoder.onnx \
          --saveEngine=/repo/vision_encoder/1/model.plan \
          --minShapes=images:1x3x1008x1008 \
          --optShapes=images:2x3x1008x1008 \
          --maxShapes=images:4x3x1008x1008 \
          --fp16

  echo 'Compiling Text Encoder...'
  trtexec --onnx=/models/text-encoder.onnx \
          --saveEngine=/repo/text_encoder/1/model.plan \
          --minShapes=input_ids:1x32,attention_mask:1x32 \
          --optShapes=input_ids:2x32,attention_mask:2x32 \
          --maxShapes=input_ids:4x32,attention_mask:4x32 \
          --fp16

  echo 'Compiling Mask Decoder...'
  trtexec --onnx=/models/decoder.onnx \
          --saveEngine=/repo/mask_decoder/1/model.plan \
          --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x32x256,prompt_mask:1x32 \
          --optShapes=fpn_feat_0:2x256x288x288,fpn_feat_1:2x256x144x144,fpn_feat_2:2x256x72x72,fpn_pos_2:2x256x72x72,prompt_features:2x32x256,prompt_mask:2x32 \
          --maxShapes=fpn_feat_0:4x256x288x288,fpn_feat_1:4x256x144x144,fpn_feat_2:4x256x72x72,fpn_pos_2:4x256x72x72,prompt_features:4x32x256,prompt_mask:4x32 \
          --fp16
"

echo "✅ Compilation complete! Compiled plan files are stored in model_repository."