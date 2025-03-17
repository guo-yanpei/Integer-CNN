# Integer-Only Inference for Deep Learning in Native C
This code refers to https://github.com/benja263/Integer-Only-Inference-for-Deep-Learning-in-Native-C.

Dowload CNN weights:
```
pip install gdown
gdown 1cvhUjKhGE6W68O1yHIjzYdq3nwKTdVOu
```

Run the code for quantized inference.
```
bash do.sh
```

The model is defined in `neural_nets.py`. The architecture also has to be specified in `convnet.cpp`.