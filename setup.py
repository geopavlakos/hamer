from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='hamer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'torch',
        'torchvision',
        'yacs',
        'mmcv==1.3.9',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
    ], 
)
