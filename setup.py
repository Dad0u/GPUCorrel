import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpucorrel", # Replace with your own username
    version="0.2",
    author="Victor Couty",
    author_email="victor.couty@gmail.com",
    description="A CUDA accelerated DIC module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LaboratoireMecaniqueLille/GPUCorrel",
    packages=['gpucorrel'],
    package_dir={'gpucorrel':'src/gpucorrel'},
    data_files=[('kernels',['src/gpucorrel/kernels/kernels.cu'])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv2",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
