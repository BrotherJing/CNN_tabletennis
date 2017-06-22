# CNN_tablentennis

Implement with the idea in SO-DLT([Transferring Rich Feature Hierarchies for Robust Visual Tracking](https://arxiv.org/abs/1501.04587)).

And also Faster R-CNN([Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497))

# Installation

1. Clone this repo with the submodule, which is our fork from caffe.
```bash
git clone --recursive <this repo>
```
2. `cd` to the customized caffe directory, install caffe using cmake, in order for our cpp interface to find the install path of caffe.
```bash
cd caffe
mkdir build
cd build
cmake ..
make all
make install
make runtest
```
3. Download our caffe model weights.

SO-CNN layer | Regression layer
----|----
[end2end_vgg_iter_20000.caffemodel](http://or9ajn83e.bkt.clouddn.com/tracking/end2end_vgg_iter_20000.caffemodel)|[end2end_regress_vgg_iter_12000.caffemodel](http://or9ajn83e.bkt.clouddn.com/tracking/end2end_regress_vgg_iter_12000.caffemodel)

4. Run the jupyter notebook, or the cpp interface:
```bash
mkdir build && cd build
cmake ..
make
```
The `lib/` directory contains the python layers, add it to `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=$PYTHONPATH:<this repo>/lib
```
5. To create dataset and train model yourself, see the shell scripts in `scripts/`.

# Screenshots

See more detail in the jupyter notebook :)

output probability map

![probability map](http://7xrcar.com1.z0.glb.clouddn.com/prob_map.png)

regress a bounding box

![bounding box](http://7xrcar.com1.z0.glb.clouddn.com/regression.png)

tracking

![tracking](http://7xrcar.com1.z0.glb.clouddn.com/track_vis.png)
