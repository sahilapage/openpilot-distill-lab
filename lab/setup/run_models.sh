#!/usr/bin/env bash
cd /home/sra/sahil_distill/openpilot.distill
python3 ../download_ranged.py https://media.githubusercontent.com/media/commaai/openpilot/084747c75d2cbd23af65ab7a9e770bbd7b98bac9/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx models/big_driving_supercombo.onnx 8
python3 ../download_ranged.py https://huggingface.co/commaai/worldmodel-4B/resolve/main/model.fp8_nvfp4.torchpackage models/model.fp8_nvfp4.torchpackage 8
echo MODELS_DONE
