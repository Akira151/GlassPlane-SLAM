# GlassPlane-SLAM

This is the official implementation of [GlassPlane-SLAM: Glass-Plane-Aware RGB-D SLAM for Improved Localization in Glass Environments](https://www.mdpi.com/1424-8220/26/21/7050), published in *Sensors*.

GlassPlane-SLAM extends ORB-SLAM3 with glass-plane mapping and plane constraints in local bundle adjustment. The code is based on [ORB_SLAM3_detailed_comments](https://github.com/electech6/ORB_SLAM3_detailed_comments), an annotated version of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).

# 1. Prerequisites

The dependencies are the same as [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3#2-prerequisites), plus **PCL, fmt and spdlog**. Install the additional dependencies with:

```bash
sudo apt install libpcl-dev libfmt-dev libspdlog-dev
```

### Development environment

The project was tested with the following library versions:

OpenCV 4.5.4, Eigen 3.4.0, Pangolin 0.5, PCL 1.13.1.

# 2. Building

```bash
git clone https://github.com/Akira151/GlassPlane-SLAM.git
cd GlassPlane-SLAM
bash build.sh
```

# 3. Datasets

- Glass dataset: Coming soon.
- Preprocessing code: Coming soon.

# 4. Data preparation

Before running SLAM:

1. Run glass segmentation on the RGB images to obtain glass masks.
2. Run our preprocessing code with the RGB images, depth maps and masks to prepare the inputs required by GlassPlane-SLAM.

Keep `associations.txt` in the sequence directory.

# 5. Running

For our dataset, use [ORBSLAM3_d455f.yaml](ORBSLAM3_d455f.yaml) in the repository root. After preparing the inputs above, run from the repository root:

```bash
./Examples/RGB-D/rgbd_tum_plane Vocabulary/ORBvoc.txt ORBSLAM3_d455f.yaml \
    PATH_TO_SEQUENCE PATH_TO_SEQUENCE/associations.txt
```

# 6. License and citation

This project is released under the [GPLv3 license](LICENSE). Third-party license information is listed in the [ORB-SLAM3 dependency notes](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Dependencies.md).

We thank the authors of ORB-SLAM3 and ORB_SLAM3_detailed_comments. If you use this software in academic work, please cite [our paper](https://www.mdpi.com/1424-8220/26/21/7050) and ORB-SLAM3:

```bibtex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial
         and Multi-Map {SLAM}},
  author={Campos, Carlos and Elvira, Richard and G{\'o}mez, Juan J. and
          Montiel, Jos{\'e} M. M. and Tard{\'o}s, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={6},
  pages={1874--1890},
  year={2021}
}
```
