# Tree Geo-localization from Google Street View
Automated tree localization from Google Street View imagery using monocular depth estimation and geometric projection methods. This repository implements both baseline method and specialized approaches for trunk-occluded scenarios.

## Project Structure

```
trees-geo-localization/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                           # Data directory
│   ├── depth-maps-visualization/   # Disparity visualization outputs
│   ├── disparities/               # Disparity .npy files from monodepth
│   ├── images/                    # Source GSV images (400x400)
│   ├── images-with-bboxes/        # Images with bbox annotations (optional)
│   └── trees-data/                # Tree detection results (CSV/Excel)
├── scale/
│   └── depth_scale.json          # Calibration scale for disparity-to-distance
└── src/
    ├── baseline/                 # V1: Standard visible-trunk pipeline
    │   ├── __init__.py
    │   ├── calibrate_depth.py    # Calibration utilities
    │   ├── pipeline_core.py      # Core processing functions
    │   └── run_pipeline.py       # Main pipeline script
    ├── trunk_occluded/           # V2-V4: Specialized methods for occluded trunks
    │   ├── enhanced_projection.py    # V2: Enhanced building-aware approach
    │   ├── simplified_projection.py  # V3: Simplified building-aware approach
    │   ├── optimized_projection.py   # V4: Optimized fixed projection
    │   └── mixture_of_experts.py     # MoE: Failed ensemble approach
    └── utils/                    # Utility scripts
        ├── monodepth_batch.py    # Batch disparity generation
        └── visualize_bboxes.py   # Bbox visualization tool
```


## Quick Start

### 1. Prepare Your Data

**Required files:**
- **Images**: GSV images
- **Bounding boxes**: CSV/Excel file with columns:
  - `file_name`: Image filename
  - `x_box`, `y_box`: Bbox center (normalized 0-1)
  - `width_box`, `height_box`: Bbox dimensions (normalized 0-1)
- **Disparity maps**: Generate using monodepth (see below)
- **Scale calibration**: `scale/depth_scale.json` (see calibration section)


### 2. Generate Disparity Maps

Disparity maps are generated using the [monodepth](https://github.com/mrharicot/monodepth) neural network.

**Setup monodepth environment:**
```bash
# Create separate conda environment (required for TensorFlow 1.x)
conda create -n monodepth python=3.7
conda activate monodepth
pip install tensorflow==1.15.0 matplotlib pillow

# Clone and setup monodepth repository
git clone https://github.com/mrharicot/monodepth.git
cd monodepth

# Download pre-trained model (model_kitti)
sh ./utils/get_model.sh model_kitti models/
```

**Generate disparity maps:**

*Option 1: Single image (using original monodepth script):*
```bash
python monodepth_simple.py --image_path your_image.jpg --checkpoint_path models/model_kitti
```

*Option 2: Batch processing (using our utility):*
```bash
# Copy our batch script to monodepth root directory (same location as monodepth_simple.py)
cp src/utils/monodepth_batch.py /path/to/monodepth/
cd /path/to/monodepth/

# Run batch processing
python monodepth_batch.py --images_dir /path/to/images --checkpoint_path models/model_kitti --output_dir /path/to/disparities
```


### 3. Scale Calibration (Optional)

**Note: A pre-calibrated scale is already provided in `scale/depth_scale.json`. You can skip this step and use the existing calibration.**

If you want to create your own calibration, edit the coordinates in `src/baseline/calibrate_depth.py`:

```python
# Edit these values in calibrate_depth.py
camera_coords = (32.05332566340407, 34.81022952939507)  # Your camera GPS
tree_coords = (32.0532986, 34.8101803)                   # Your reference tree GPS  
disp_avg = 0.034628998                                    # Average disparity from tree bbox

# Run calibration
python src/baseline/calibrate_depth.py
```

This will create a new `depth_scale.json` file with your custom calibration.


### 4. Run Tree Localization

## Baseline Pipeline (V1) - For Visible Trunk Trees

The baseline pipeline can process either single images or entire batches. It automatically handles multiple trees per image by processing each bounding box row in the CSV file.

**Batch processing (recommended):**
```bash
python -m src.baseline.run_pipeline batch \
    --excel data/trees-data/your_annotations.csv \
    --disp_dir data/disparities \
    --output_csv results/all_trees.csv
```

**Single image:**
```bash
python -m src.baseline.run_pipeline single \
    --excel data/trees-data/your_annotations.csv \
    --image_name your_image.jpg \
    --output_csv results/single_result.csv
```

**How it handles multiple trees:** The pipeline reads each row in the CSV file as a separate tree. If one image has multiple trees, there will be multiple rows with the same `file_name` but different bounding box coordinates. The pipeline processes each row independently and outputs all tree locations.

## Trunk-Occluded Methods (V2-V4) - For Occluded Trees

These methods process **one tree at a time** (not entire images) because they require manual selection of which trees need the specialized approach. Each method takes a `tree_index` parameter to specify which bounding box within the image to process.

**V2 - Enhanced Building-Aware:**
```bash
python src/trunk_occluded/enhanced_projection.py \
    --csv_path data/trees-data/your_annotations.csv \
    --image_name "your_image.jpg" \
    --tree_index 0 \
    --disp_path data/disparities \
    --scale_path scale/depth_scale.json
```

**V3 - Simplified Building-Aware:**
```bash
python src/trunk_occluded/simplified_projection.py \
    --csv_path data/trees-data/your_annotations.csv \
    --image_name "your_image.jpg" \
    --tree_index 0 \
    --disp_path data/disparities \
    --scale_path scale/depth_scale.json
```

**V4 - Optimized Fixed Projection:**
```bash
python src/trunk_occluded/optimized_projection.py \
    --csv_path data/trees-data/your_annotations.csv \
    --image_name "your_image.jpg" \
    --tree_index 0 \
    --disp_path data/disparities \
    --scale_path scale/depth_scale.json
```


**Understanding tree_index:** If your CSV has multiple rows for the same image (multiple trees), use:
- `--tree_index 0` for the first tree in that image
- `--tree_index 1` for the second tree in that image
- And so on...

**Note:** Currently, you must manually decide which method to use for each tree. Future work might include developing an automatic classifier to route trees to the appropriate pipeline.


## Utility Scripts

### Visualize Bounding Boxes
```bash
python src/utils/visualize_bboxes.py \
    --images_dir data/images \
    --table data/trees-data/your_annotations.csv \
    --out_dir data/images-with-bboxes
```

### Batch Disparity Generation
For generating disparity maps from multiple images using monodepth:

```bash
# Copy our batch script to monodepth root directory (same location as monodepth_simple.py)
cp src/utils/monodepth_batch.py /path/to/monodepth/
cd /path/to/monodepth/

# Run batch processing
python monodepth_batch.py --images_dir /path/to/images --checkpoint_path models/model_kitti --output_dir /path/to/disparities
```

**Note:** The `monodepth_batch.py` script is our custom extension of the original monodepth for processing multiple images efficiently.




