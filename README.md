# SuperPoint

This is a Tensorflow implementation of  "SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiv 2018](https://arxiv.org/abs/1712.07629).

![hp-v_200](doc/hp-v_200.png)
![hp-v_235](doc/hp-v_235.png)
![hp-v_280](doc/hp-v_280.png)

## HPatches repeatability results

Repeatability on HPatches computed with 300 points detected in common between pairs of images and with a NMS of 4:
 <table style="width:100%">
  <tr>
    <th></th>
    <th>Illumination changes</th>
    <th>Viewpoint changes</th>
  </tr>
  <tr>
    <td>SuperPoint (our implementation)</td>
    <td><b>0.661</b></td>
    <td>0.409</td>
  </tr>
  <tr>
    <td>SuperPoint (<a href='https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork' >pretrained model of MagicLeap<a>)</td>
    <td>0.641</td>
    <td>0.379</td>
  </tr>
  <tr>
    <td>FAST</td>
    <td>0.577</td>
    <td>0.415</td>
  </tr>
  <tr>
    <td>Harris</td>
    <td>0.630</td>
    <td><b>0.474</b></td>
  </tr>
  <tr>
    <td>Shi</td>
    <td>0.583</td>
    <td>0.407</td>
</table>

**NOTE**: this work is still in progress. Although we match the results of the original paper for illumination changes, more experiments are required to 1) improve the robustness to viewpoint changes, and 2) train the descriptor. **We provide a temporary Tensorflow SavedModel of MagicPoint in `superpoint/saved_models/`.**

## Installation

```shell
make install  # install the Python requirements and setup the paths
```
Python 3.6.1 is required. You will be asked to provide a path to an experiment directory (containing the training and prediction outputs, referred as `$EXPER_DIR`) and a dataset directory (referred as `$DATA_DIR`). Create them wherever you wish and make sure to provide their absolute paths.

MS-COCO 2014 and HPatches should be downloaded into `$DATA_DIR`. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:
```
$DATA_DIR
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
```

## Usage
All commands should be executed within the `superpoint/` subfolder. When training a model or exporting its predictions, you will often have to change the relevant configuration file in `superpoint/configs/`. Both multi-GPU training and export are supported.

### 1) Training MagicPoint on Synthetic Shapes
```
python experiment.py train configs/magic-point_shapes.yaml magic-point_synth
```
where `magic-point_synth` is the experiment name, which may be changed to anything. The training can be interrupted at any time using `Ctrl+C` and the weights will be saved in `$EXPER_DIR/magic-point_synth/`. The Tensorboard summaries are also dumped there. When training for the first time, the Synthetic Shapes dataset will be generated.

### 2) Exporting detections on MS-COCO

```
python export_detections.py configs/magic-point_coco_export.yaml magic-point_synth --pred_only --batch_size=5 --export_name=magic-point_coco-export1
```
This will save the pseudo-ground truth interest point labels to `$EXPER_DIR/outputs/magic-point_coco-export1/`. You might enable or disable the Homographic Adaptation in the configuration file.

### 3) Training MagicPoint on MS-COCO
```
python experiment.py train configs/magic-point_coco_train.yaml magic-point_coco
```
You will need to indicate the paths to the interest point labels in `magic-point_coco_train.yaml` by setting the entry `data/labels`, for example to `outputs/magic-point_coco-export1`. You might repeat steps 2) and 3) several times.

### 4) Evaluating the repeatability on HPatches
```
python export_detections_repeatability.py configs/magic-point_repeatability.yaml magic-point_coco --export_name=magic-point_hpatches-repeatability-v
```
You will need to decide whether you want to evaluate for viewpoint or illumination by setting the entry `data/alteration` in the configuration file. The predictions of the image pairs will be saved in `$EXPER_DIR/outputs/magic-point_hpatches-repeatability-v/`. To proceed to the evaluation, head over to `notebooks/detector_repeatability_coco.ipynb`. You can also evaluate the repeatbility of the classical detectors using the configuration file `classical-detectors_repeatability.yaml`.

### 5) Validation on MS-COCO
It is also possible to evaluate the repeatability on a validation split of COCO. You will first need to generate warped image pairs using `generate_coco_patches.py`.

## Credits
This implementation was developed by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes). Please contact Rémi for any enquiry.
