from pathlib import Path

# Path to the root of the project
HERE = Path(__file__).parent


# Path to the data directory

SAMPLE_DOCUMENTS_DIR = HERE / "Sample documents/"
SwinDocSegmenter_DIR = HERE / "SwinDocSegmenter"
SwinDocSegmenter_DIR_WEIGHTS = SwinDocSegmenter_DIR / "weights"
SwinDocSegmenter_DIR_CONFIGS = SwinDocSegmenter_DIR / "configs" / "coco" / "instance-segmentation" / "swin"

# Weights 
WEIGHTS_PATH_prima     = SwinDocSegmenter_DIR_WEIGHTS / 'model_final_prima_swindocseg.pth'
WEIGHTS_PATH_doclaynet = SwinDocSegmenter_DIR_WEIGHTS / 'model_final_doclay_swindocseg.pth'
WEIGHTS_PATH_publaynet = SwinDocSegmenter_DIR_WEIGHTS / 'model_final_publay_swindocseg.pth'
WEIGHTS_PATH_tablebank = SwinDocSegmenter_DIR_WEIGHTS / 'model_final_table_swindocseg.pth'

# CONFIGS
CONFIG_PATH_prima     = SwinDocSegmenter_DIR_CONFIGS / "config_prima.yaml"
CONFIG_PATH_doclaynet = SwinDocSegmenter_DIR_CONFIGS / "config_doclay.yaml"
CONFIG_PATH_publaynet = SwinDocSegmenter_DIR_CONFIGS / "config_publay.yaml"
CONFIG_PATH_tablebank = SwinDocSegmenter_DIR_CONFIGS / "config_table.yaml"