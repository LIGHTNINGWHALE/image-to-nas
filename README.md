## NIO Student Hackathon

This is the complete code for generating. nas files submitted by our group.
When it is run reads the images in `jpeg images/` and writes the generated `.nas` files to `submission_nas/`.


## Team
#@team LZZ, WXY, JZY 
#@date 2026-04-14


## Our running environment

- 13th Gen Intel(R) Core(TM) i7-13700H
- NVIDIA GeForce RTX 3050 4GB Laptop GPU
- windows 11

- torch_version 2.11.0+cu128
- cuda_version 12.8
- python 14
  - numpy
  - scipy
  - matplotlib
  - Pillow
  - opencv-python-headless
  - scikit-image
  - ortools
  - shapely
  - torch
  - torchvision
  - yacs
  - tqdm
  - easydict
  - python-json-logger
  - PyYAML


## Directory Structure

- `main.py`: single entrypoint used to reproduce outputs
- `jpeg images/`: the 100 competition input images
- `submission_nas/`: final generated `.nas` files
- `debug_output/`: full debug artifacts, only created when `--debug` is used
- `Example .nas Format/`: two example image/NAS pairs used for offline validation
- `nio_mesh/`: pipeline implementation
- `models/checkpoints/`: bundled HAWPv3 checkpoint
- `requirements_submission.txt`: minimal runtime dependencies for this submission bundle
- `Report.pdf`: one-page report


## Install Dependencies

From this directory, install the runtime requirements first:

```
pip install -r requirements_submission.txt
```

## Default Reproduction

Then run:

```
python main.py
```

This will:

- process all images in `jpeg images/`
- generate the final 100 `.nas` files
- write them into `submission_nas/`
- write the element `file_id` as the image number for each file, for example `hole_001 -> 1` and `hole_100 -> 100`

We recommend running on GPU, This way, the speed can be increased several times

```
python main.py --device cuda
```

if can not find GPU, you need check your cuda, If the following code cannot solve the problem, it is recommended to consult AI.

```
nvidia-smi
```

```
python -m pip uninstall -y torch torchvision
python -m pip cache purge
python -m pip install --no-cache-dir --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## Debug Reproduction

To keep the full debug output:

```
python main.py --debug
```

This will:

- generate the same final `.nas` files in `submission_nas/`
- keep the full intermediate tree in `debug_output/`

## Benchmark Examples

To run the two example benchmark images and generate comparison reports:

```
python main.py --benchmark-examples
```

This will write outputs into `benchmark_output/`.

## Optional Arguments

- `--input-dir <dir>`: alternate image directory
- `--output-dir <dir>`:
  - normal mode: alternate final `.nas` directory
  - debug mode: alternate debug artifact root
  - benchmark mode: alternate benchmark output root
- `--images <paths...>`: explicit image paths to run
- `--image-names <names...>`: filenames under `--input-dir` to run
- `--device auto|cuda|cpu`: execution device preference

## Code Documentation

The code is documented with a clear separation of responsibilities so the `.nas` generation process can be reproduced directly from this bundle.

- `main.py` is the single entrypoint for reproduction.

The main submission logic is implemented in `nio_mesh/`. The role of each file is:

- `pipeline.py`: the top-level pipeline controller. It loads each image, runs preprocessing, launches HAWP, builds reconstruction candidates, applies filtering, and writes the final outputs.
- `axis.py`: detects the magenta axes, estimates the origin and axis spans, stabilizes rare axis-detection failures, and provides the pixel-to-world mapping used for final coordinates.
- `hawp_bundle.py`: the submission-bundle HAWP entrypoint. It runs the fixed HAWPv3 sweep used in this bundle and converts raw HAWP outputs into the format needed by reconstruction.
- `vendor_hawp_runner.py`: resolves the execution device and calls the bundled HAWPv3 inference runtime in `nio_mesh/vendor_hawp/`.
- `hawp.py`: shared HAWP helper utilities, including HAWP config definitions, input-variant generation, graph remapping, segment export, overlay rendering, and cache-compatible helper logic.
- `reconstruct.py`: the core mesh-recovery file. It converts processed wireframes into mesh candidates, performs graph cleanup, traces faces, applies marker-aware fixes, polygonizes bounded regions, and assembles the final mesh topology.
- `element_filter.py`: removes reconstructed elements whose centroid region falls on black background in the original image.
- `coordinate_correction.py`: applies the fixed coordinate correction layer learned offline from the two example benchmark image/NAS pairs.
- `geometry.py`: geometry utilities such as clustering, polygon ordering, line fitting, edge collapse, and other reusable geometric operations.
- `mesh_solver.py`: contains constraint-based mesh selection and refinement utilities used to assemble valid final candidates.
- `mesh_validator.py`: validates final mesh candidates and checks for issues such as intersections, invalid polygons, and bad connectivity.
- `nas.py`: defines the in-memory NAS mesh structures, parses `.nas` files, and writes final `.nas` outputs in the competition format.
- `compare.py`: benchmark-only comparison logic for the two example image/NAS pairs. It is used for evaluation, not for normal submission generation.
- `deepvec.py`: legacy alternate line-detection support kept in the codebase, but not used by the fixed submission configuration.
- `__init__.py`: package marker for `nio_mesh`.

## Runtime Logic

The runtime flow is:

1. `main.py` collects the requested images and calls `nio_mesh.pipeline.run_full_mode`.
2. `pipeline.py` loads each image and calls `axis.py` to detect and debug the magenta coordinate system.
3. The same pipeline removes axis artifacts, isolates the mesh region, and creates the fixed HAWP input variants.
4. `hawp_bundle.py` calls `vendor_hawp_runner.py`, which imports the bundled HAWPv3 code and runs inference.
5. The raw HAWP graph is remapped into full-image coordinates and passed into `reconstruct.py`.
6. `reconstruct.py` cleans the graph, traces bounded regions, generates mesh candidates, and converts the selected candidate into a NAS mesh object.
7. `element_filter.py` removes faces that lie on black background holes.
8. `axis.py` and `coordinate_correction.py` convert pixel coordinates into final world coordinates.
9. `nas.py` writes the final `.nas` file into `submission_nas/`.


## Other

AI-tool disclosure is recorded in report.

The vendored HAWP runtime is included under the MIT license. See:

## HAWP Bundle Changes

Source: local `hawp-main` checkout bundled from the MIT-licensed HAWP project.

This submission bundle vendors only the HAWPv3 inference runtime needed by the stable competition pipeline.

Applied reductions and modifications:

- Removed items that were not useful for the project
- removed HAWPv2-specific runtime packaging
- vendored the remaining HAWPv3 inference code under `nio_mesh/vendor_hawp/`
- replaced `hawp.ssl.predict` with a submission-specific simplified inference entrypoint
- disabled runtime JIT compilation of the optional HAWP C++/CUDA extension by forcing `_C = None`
- trimmed `hawp.base` to the overlay/wireframe functionality needed by the stable pipeline
- kept the model/backbone/config code required by the HAWPv3 checkpoint used in this submission

These changes were made solely to simplify packaging and reproducibility for the competition submission bundle.

# MIT License

Copyright (c) 2023 Nan Xue

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
