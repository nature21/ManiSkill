from setuptools import setup, find_packages

setup(
    name="diffusion_policy_3d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "wandb",
        "mani_skill"
    ],
    description="A minimal setup for 3D Diffusion Policy for ManiSkill",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
