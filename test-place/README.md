# How to combine pytorch, c++ and CUDA codes

You can see [this example](./b.py) with [cpp code](./lltm.cpp) as the combination of pytorch and cpp.

You can also see [this example](./c.py) with [cpp code](./lltm_cu.cpp) and [cu code](./lltm.cu) as the combination of pytoch, cpp and cuda.

These codes are from [the pytorch tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html#), and is verified on my computer(Windows11, CUDA 12.1, pytorch, g++-11.2.0). If you want to try the tutorial yourself, I highly recommend you to check ABI-compatible first. 