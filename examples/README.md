# Interactive Video Stylization Using Few-Shot Patch-Based Training

This is a demo that showcases the ability of the algorithm proposed in 
this SIGGRAPH 2020 paper: [Interactive Video Stylization Using Few-Shot Patch-Based Training](https://ondrejtexler.github.io/res/Texler20-SIG_patch-based_training_main.pdf)
to transfer the style generated from Stable Diffusion for a few keyframes to the rest of the video.

## Configuration
The official Github repository is available [here](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training).
You are supposed to clone this repository to this directory:
```shell
git clone https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training.git
```

You will have make some minor changes to the code to make it work as Tensorflow 1.15 is not supported anymore
and Pytorch has already has TensorBoard integrated. Specifically, you'll need to replace the `Logger` class in logger.py
to the following equivalent code:
```python
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        """Create a summary writer logging to log_dir."""
        if suffix is None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = SummaryWriter(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
```
After that, run the following command to install the dependencies:
```shell
pip install -r requirements.txt
```
Then you are good to go.

## Preprocessing

You should arrange your data in the following structure:
```
── {project_name}
   ├── {process_name}_gen
   │   ├── input_filtered
   │   └── whole_video_input
   └── {process_name}_train
       ├── input_filtered
       └── output
```

* {projectName} : The name of your project, e.g. "woman_dance".
* {process_name}_gen/input_filtered : Put raw frames that are not keyframes, 10 is enough to observe the effect during training.
* {process_name}_gen/whole_video_input : Put all raw video frames. It is used to final generation.
* {process_name}_train/input_filtered : Put raw keyframe images.
* {process_name}_train/output : Put generated keyframe images by stable diffusion's image-to-image. Image names should be the same as input_filtered.

You can also have some mask to control the part of the image that will be stylized, but for this demo we simply leave them empty by running:
```shell
python gen_empty_masks.py
```
The scripts to convert the video to cropped frames can be found in preprocessing folder.

## Training

To train the model, run the following command:
```shell
$ cd Few-Shot-Patch-Based-Training
$ python train.py --config "_config/reference_P.yaml" --data_root {train_dir} --log_interval 2000 --log_folder logs_reference_P
```
where {train_dir} is the path to the training data directory. You can also change the configuration file to train the model with different parameters.
