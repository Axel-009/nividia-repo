# TensorFlow-TensorRT (TF-TRT) Image Classification on Google Colab

## Overview
The TensorFlow-TensorRT (TF-TRT) examples in the [official repository](https://github.com/tensorflow/tensorrt/tree/master/tftrt/benchmarking-python/image_classification) are outdated and no longer function correctly with newer TensorFlow versions.

This README provides updated information and a working version of the Colab notebook for TF-TRT integration using TensorFlow 2.17.0.

## Affected Notebooks
The following notebooks fail to run on TensorFlow 2.1+ due to TensorRT compatibility issues:

- [Colab-TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb](https://github.com/tensorflow/tensorrt/blob/master/tftrt/benchmarking-python/image_classification/Colab-TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb)
- [NGC-TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb](https://github.com/tensorflow/tensorrt/blob/master/tftrt/benchmarking-python/image_classification/NGC-TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb)

## Problem Description
These notebooks were originally designed for TensorFlow 2.0.0, but they fail with TensorFlow 2.1+ with the following error:

```
RuntimeError: TensorFlow has not been built with TensorRT support.
```

This issue has been discussed in the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/runtimeerror-tensorflow-has-not-been-built-with-tensorrt-support/238124/9), but no official resolution has been provided.

## Findings & Fix
- TF-TRT was tested on Google Colab with TensorFlow 2.17.0, along with compatible CUDA and TensorRT versions.
- It works correctly with TensorFlow 2.17.0 after some minor modifications.
- TensorFlow 2.18.0 does not support TensorRT, making 2.17.0 the latest version with full compatibility.

### Fix for Newer TensorFlow Versions
For later TensorFlow versions that include TensorRT support, ensure that the correct TensorRT runtime libraries are installed. For example, in TensorFlow 2.17.0, the following command installs the required libraries:

```
sudo apt-get install -y libnvinfer8 libnvinfer-plugin8
```

## Proposed Contribution
A working version of the Colab notebook has been created to successfully run TF-TRT on TensorFlow 2.17.0. This will assist developers and researchers in overcoming compatibility issues.

## Next Steps
- Update the official documentation to reflect the correct TensorFlow, CUDA, and TensorRT versions that work together.
- Consider maintaining an up-to-date compatibility matrix for future versions.

## References
- [TensorFlow-TensorRT Repository](https://github.com/tensorflow/tensorrt/tree/master/tftrt/benchmarking-python/image_classification)
- [NVIDIA Developer Forum Discussion](https://forums.developer.nvidia.com/t/runtimeerror-tensorflow-has-not-been-built-with-tensorrt-support/238124/9)

