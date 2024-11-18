# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

Training Results:
GPU Split 13m36s -> 1.632s per epoch
Epoch  0  loss  7.085095862969477 correct 33
Epoch  10  loss  5.135104696648748 correct 33
Epoch  20  loss  6.518139810770241 correct 42
Epoch  30  loss  5.2011862472881525 correct 47
Epoch  40  loss  3.5827199574099566 correct 47
Epoch  50  loss  6.337727919386756 correct 45
Epoch  60  loss  2.0787647244026513 correct 49
Epoch  70  loss  4.010110289345822 correct 49
Epoch  80  loss  1.1853478925732013 correct 49
Epoch  90  loss  1.3373527666614486 correct 48
Epoch  100  loss  1.420274393170151 correct 49
Epoch  110  loss  1.400832170883739 correct 50
Epoch  120  loss  0.4797797423042985 correct 50
Epoch  130  loss  0.8309820324994397 correct 48
Epoch  140  loss  1.7176904610402124 correct 50
Epoch  150  loss  1.0752130931880197 correct 49
Epoch  160  loss  0.42825397687245215 correct 50
Epoch  170  loss  1.4016333889757688 correct 50
Epoch  180  loss  1.0684721832374002 correct 50
Epoch  190  loss  0.9419520020843017 correct 50
Epoch  200  loss  0.308772005563778 correct 50
Epoch  210  loss  0.47033894463316617 correct 50
Epoch  220  loss  0.6375476452073842 correct 50
Epoch  230  loss  0.5108575220154032 correct 50
Epoch  240  loss  0.9063251249690607 correct 50
Epoch  250  loss  0.29657137646395026 correct 50
Epoch  260  loss  0.10434651898264809 correct 50
Epoch  270  loss  0.22237293351468249 correct 50
Epoch  280  loss  0.19043870661348175 correct 50
Epoch  290  loss  0.7487420404758215 correct 50
Epoch  300  loss  0.2125774592790732 correct 50
Epoch  310  loss  0.400555647296219 correct 50
Epoch  320  loss  0.2052023523173438 correct 50
Epoch  330  loss  0.5845826592743638 correct 50
Epoch  340  loss  0.20456903730430825 correct 50
Epoch  350  loss  0.06784766604389024 correct 50
Epoch  360  loss  0.39658193769532485 correct 50
Epoch  370  loss  0.09597088316237984 correct 50
Epoch  380  loss  0.3618077826554561 correct 50
Epoch  390  loss  0.1254851103955751 correct 50
Epoch  400  loss  0.40086413723787484 correct 50
Epoch  410  loss  0.059583535460922196 correct 50
Epoch  420  loss  0.14088995351287092 correct 50
Epoch  430  loss  0.16648289921325554 correct 50
Epoch  440  loss  0.42246063354541985 correct 50
Epoch  450  loss  0.28021940653003347 correct 50
Epoch  460  loss  0.08202046998260529 correct 50
Epoch  470  loss  0.2691160274507436 correct 50
Epoch  480  loss  0.1012465868382427 correct 50
Epoch  490  loss  0.02662817361865362 correct 50


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py