#python3 create_convnet_c_params.py
g++ -Wall -fPIC -c convnet_params.cpp convnet.cpp nn_math.cpp nn.cpp  
g++ -shared convnet_params.o convnet.o nn_math.o nn.o -o convnet.so
python3 test_convnet_c.py