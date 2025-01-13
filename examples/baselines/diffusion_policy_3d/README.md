# 3D Diffusion Policy

Code for running the 3D Diffusion Policy algorithm based on ["3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations"]. It is adapted from the [original code](https://github.com/YanjieZe/3D-Diffusion-Policy).

## Installation


```bash
conda create -n maniskill-dp3 python=3.9
conda activate maniskill-dp3
cd ~/ManiSkill/examples/baselines/diffusion_policy_3d
pip install -e .
pip install torch torchvision torchaudio

cd
# install pytorch3d for poindcloud downsampling
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . # this can take more than 10 mins
```

## Demonstration Preparation and Preprocessing


```bash
conda activate maniskill-dp3
cd ~/ManiSkill

python mani_skill/examples/motionplanning/panda/run.py -e "PickCube-v1" --traj-name="trajectory_cpu" -n 10 --record-dir "~/.maniskill/demos/PickCube-v1/motionplanning/" --sim-backend "cpu" --save-video --only-count-success
```

```bash
python -m mani_skill.trajectory.replay_trajectory --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory_cpu.h5 --use-first-env-state -c pd_ee_delta_pose -o pointcloud --save-traj --num-procs 16
```

Convert ManiSkill hdf5 dataset into zarr format for DP3 training:
```bash
cd ~/ManiSkill/examples/baselines/diffusion_policy_3d

python dataset/convert_hdf5_to_zarr.py --num_demos 10 --zarr_dir ./data/ --hdf5_path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory_cpu.pointcloud.pd_ee_delta_pose.cpu.h5 --env_name PickCube-v1
```

## Training

```bash
conda activate maniskill-dp3
cd ~/ManiSkill/examples/baselines/diffusion_policy_3d

# 1) argument 1: algorithm name {dp3, simple_dp3}
# 2) argument 2: task name
# 3) argument 3: addition info, i.e., date
# 4) argument 4: seed
# 5) argument 5: gpu id
bash train_policy.sh simple_dp3 PickCube-v1 0114 0 0
```

## Citation

If you use this baseline please cite the following
```
@inproceedings{ze20243d,
  title={3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations},
  author={Ze, Yanjie and Zhang, Gu and Zhang, Kangning and Hu, Chenyuan and Wang, Muhan and Xu, Huazhe},
  booktitle={ICRA 2024 Workshop on 3D Visual Representations for Robot Manipulation},
  year={2024}
}
```