# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

## Task 3.1, 3.2 NUMBA diagnostics
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        # TODO: Implement for Task 3.1.                                      | 
        out_size = len(out)                                                  | 
        if np.array_equal(out_shape, in_shape) and np.array_equal(           | 
            out_strides, in_strides                                          | 
        ):                                                                   | 
            for ordinal in prange(out_size):---------------------------------| #2
                # If stride-aligned, avoid indexing                          | 
                out[ordinal] = fn(float(in_storage[ordinal]))                | 
        else:                                                                | 
            for ordinal in prange(out_size):---------------------------------| #3
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------| #0
                in_index = np.zeros(MAX_DIMS, dtype=np.int32)----------------| #1
                to_index(ordinal, out_shape, out_index)                      | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                in_pos = index_to_position(in_index, in_strides)             | 
                out_pos = index_to_position(out_index, out_strides)          | 
                out[out_pos] = fn(in_storage[in_pos])                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (181) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (182) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (215)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (215) 
-------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                              | 
        out: Storage,                                                                      | 
        out_shape: Shape,                                                                  | 
        out_strides: Strides,                                                              | 
        a_storage: Storage,                                                                | 
        a_shape: Shape,                                                                    | 
        a_strides: Strides,                                                                | 
        b_storage: Storage,                                                                | 
        b_shape: Shape,                                                                    | 
        b_strides: Strides,                                                                | 
    ) -> None:                                                                             | 
        # TODO: Implement for Task 3.1.                                                    | 
        out_size = len(out)                                                                | 
        if (                                                                               | 
            np.array_equal(a_shape, b_shape)                                               | 
            and np.array_equal(b_shape, out_shape)                                         | 
            and np.array_equal(a_strides, b_strides)                                       | 
            and np.array_equal(b_strides, out_strides)                                     | 
        ):  # If stride-aligned, avoid indexing                                            | 
            for ordinal in prange(out_size):-----------------------------------------------| #7
                out[ordinal] = fn(float(a_storage[ordinal]), float(b_storage[ordinal]))    | 
        else:                                                                              | 
            for ordinal in prange(out_size):-----------------------------------------------| #8
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)-----------------------------| #4
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------------------| #5
                b_index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------------------| #6
                to_index(ordinal, out_shape, out_index)                                    | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                    | 
                a_pos = index_to_position(a_index, a_strides)                              | 
                b_pos = index_to_position(b_index, b_strides)                              | 
                out_pos = index_to_position(out_index, out_strides)                        | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (238) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (239) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (240) is 
hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (273)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (273) 
----------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                        | 
        out: Storage,                                                                   | 
        out_shape: Shape,                                                               | 
        out_strides: Strides,                                                           | 
        a_storage: Storage,                                                             | 
        a_shape: Shape,                                                                 | 
        a_strides: Strides,                                                             | 
        reduce_dim: int,                                                                | 
    ) -> None:                                                                          | 
        # TODO: Implement for Task 3.1.                                                 | 
        reduce_size = a_shape[reduce_dim]                                               | 
        for i in prange(len(out)):------------------------------------------------------| #12
            out_index = np.zeros(MAX_DIMS, np.int32)  # len(outshape)-------------------| #10
            a_index = np.zeros(MAX_DIMS, np.int32)--------------------------------------| #11
            to_index(i, out_shape, out_index)                                           | 
            o = index_to_position(out_index, out_strides)                               | 
            result = 0.0                                                                | 
            a_index[: len(out_shape)] = out_index[: len(out_shape)]---------------------| #9
            for s in range(reduce_size):                                                | 
                a_index[reduce_dim] = s  # Set the reduce dimension                     | 
                j = index_to_position(a_index, a_strides)  # Compute position in `a`    | 
                result = fn(result, a_storage[j])                                       | 
            out[o] = result                                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #12, #10, #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--12 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--10 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (parallel)
   +--10 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (serial)
   +--10 (serial)
   +--11 (serial)


 
Parallel region 0 (loop #12) had 0 loop(s) fused and 3 loop(s) serialized as 
part of the larger parallel loop (#12).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (285) is 
hoisted out of the parallel loop labelled #12 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)  # len(outshape)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (286) is 
hoisted out of the parallel loop labelled #12 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (300)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Desktop/MLE/mod3-xiaocheny1209/minitorch/fast_ops.py (300) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    assert a_shape[-1] == b_shape[-2], "Incompatible matrix shapes for multiplication"      | 
                                                                                            | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    N, I, J, K = out_shape[0], out_shape[1], out_shape[2], a_shape[-1]                      | 
    for n in prange(N):---------------------------------------------------------------------| #16
        for i in prange(I):-----------------------------------------------------------------| #15
            for j in prange(J):-------------------------------------------------------------| #14
                for k in prange(K):---------------------------------------------------------| #13
                    out_ordinal = (                                                         | 
                        n * out_strides[0] + i * out_strides[1] + j * out_strides[2]        | 
                    )                                                                       | 
                    a_ordinal = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]    | 
                    b_ordinal = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]    | 
                    out[out_ordinal] += a_storage[a_ordinal] * b_storage[b_ordinal]         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #16, #15).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--16 is a parallel loop
   +--15 --> rewritten as a serial loop
      +--14 --> rewritten as a serial loop
         +--13 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--16 (parallel)
   +--15 (parallel)
      +--14 (parallel)
         +--13 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--16 (parallel)
   +--15 (serial)
      +--14 (serial)
         +--13 (serial)


 
Parallel region 0 (loop #16) had 0 loop(s) fused and 3 loop(s) serialized as 
part of the larger parallel loop (#16).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

## Task 3.5 training logs

### CPU simple 53s/250 epochs
Epoch  0  loss  4.538700536174641 correct 32
Epoch  10  loss  1.740674133601093 correct 49
Epoch  20  loss  2.153498482474962 correct 48
Epoch  30  loss  1.3126586137234397 correct 50
Epoch  40  loss  0.781888463706353 correct 48
Epoch  50  loss  0.7882589598036325 correct 49
Epoch  60  loss  1.038618545544056 correct 50
Epoch  70  loss  0.14875939352320036 correct 49
Epoch  80  loss  0.05216746779478988 correct 50
Epoch  90  loss  0.08011535590458556 correct 50
Epoch  100  loss  0.5818372374808518 correct 49
Epoch  110  loss  0.1252805594966681 correct 49
Epoch  120  loss  0.9662309796958088 correct 50
Epoch  130  loss  0.2640211293620027 correct 49
Epoch  140  loss  0.24469878038165546 correct 50
Epoch  150  loss  0.7742738495467271 correct 49
Epoch  160  loss  0.978097289630389 correct 50
Epoch  170  loss  0.22309448065109758 correct 49
Epoch  180  loss  0.47313141016715143 correct 50
Epoch  190  loss  0.8485747109767398 correct 50
Epoch  200  loss  0.7653047242275276 correct 50
Epoch  210  loss  0.28581439958169663 correct 50
Epoch  220  loss  0.7948309976783846 correct 50
Epoch  230  loss  0.2813139986686126 correct 50
Epoch  240  loss  0.09629365577155112 correct 50

### CPU xor 53s/250 epochs
Epoch  0  loss  6.2811768886322845 correct 32
Epoch  10  loss  5.093064994944101 correct 42
Epoch  20  loss  4.208719964698893 correct 43
Epoch  30  loss  3.1886498249381763 correct 44
Epoch  40  loss  3.8952152078797755 correct 46
Epoch  50  loss  2.7209727793828797 correct 47
Epoch  60  loss  3.0850414754764066 correct 45
Epoch  70  loss  2.241702352598865 correct 46
Epoch  80  loss  1.4268137992944323 correct 48
Epoch  90  loss  2.949579842012043 correct 47
Epoch  100  loss  2.566996736838817 correct 47
Epoch  110  loss  4.1586637733516465 correct 47
Epoch  120  loss  1.9570957833149807 correct 48
Epoch  130  loss  1.3694956895488197 correct 48
Epoch  140  loss  1.5508740624141062 correct 48
Epoch  150  loss  0.7812004344381867 correct 46
Epoch  160  loss  1.0441358671446768 correct 48
Epoch  170  loss  1.7332322708486452 correct 49
Epoch  180  loss  0.8532596692902574 correct 48
Epoch  190  loss  0.31233942843034246 correct 48
Epoch  200  loss  2.323916364673306 correct 49
Epoch  210  loss  0.8801845981592649 correct 50
Epoch  220  loss  2.338583375590053 correct 48
Epoch  230  loss  0.368894462499077 correct 50
Epoch  240  loss  0.7291473864650654 correct 49

### CPU split 52s/250 epochs
Epoch  0  loss  8.568775250552353 correct 30
Epoch  10  loss  6.046916570398818 correct 37
Epoch  20  loss  6.389172778985675 correct 44
Epoch  30  loss  5.7808435559080955 correct 36
Epoch  40  loss  5.387957995212297 correct 42
Epoch  50  loss  6.508133813512665 correct 39
Epoch  60  loss  4.0194488588819315 correct 34
Epoch  70  loss  4.049975981810297 correct 41
Epoch  80  loss  3.3803446163234065 correct 42
Epoch  90  loss  9.385738390119045 correct 40
Epoch  100  loss  3.6686429713487403 correct 43
Epoch  110  loss  2.620058391403627 correct 45
Epoch  120  loss  2.196998964549536 correct 46
Epoch  130  loss  3.512331024403359 correct 43
Epoch  140  loss  3.497767292574485 correct 44
Epoch  150  loss  1.3851819494556088 correct 42
Epoch  160  loss  1.9676972157190091 correct 46
Epoch  170  loss  2.389719269025199 correct 47
Epoch  180  loss  2.115580028615186 correct 49
Epoch  190  loss  2.022258475419287 correct 43
Epoch  200  loss  2.5650520913822 correct 44
Epoch  210  loss  2.5244010658636684 correct 45
Epoch  220  loss  0.7858783545679108 correct 48
Epoch  230  loss  2.1415909385772 correct 48
Epoch  240  loss  2.532073572742573 correct 44


### GPU simple 6m54s/250 epochs -> 1.656s per epoch
Epoch  0  loss  5.994878342817337 correct 33
Epoch  10  loss  2.1885971956100105 correct 46
Epoch  20  loss  1.9918966862851495 correct 49
Epoch  30  loss  0.9521862731879969 correct 50
Epoch  40  loss  0.7262658778992891 correct 49
Epoch  50  loss  1.581927698885625 correct 50
Epoch  60  loss  1.518873750860876 correct 50
Epoch  70  loss  1.2797733833125686 correct 50
Epoch  80  loss  0.8820448790335536 correct 48
Epoch  90  loss  0.8972905321201721 correct 50
Epoch  100  loss  0.36026511831831803 correct 50
Epoch  110  loss  0.675203345541541 correct 50
Epoch  120  loss  0.5530983918412864 correct 49
Epoch  130  loss  0.47753193804866984 correct 50
Epoch  140  loss  0.2602784841137245 correct 50
Epoch  150  loss  0.16956843352142253 correct 50
Epoch  160  loss  0.46549482985350793 correct 50
Epoch  170  loss  0.6039765992348883 correct 50
Epoch  180  loss  0.33619500135733066 correct 50
Epoch  190  loss  0.42682437531026385 correct 50
Epoch  200  loss  0.12999785715508894 correct 50
Epoch  210  loss  0.41582130539168133 correct 50
Epoch  220  loss  0.24355010835757485 correct 50
Epoch  230  loss  0.3565898755194501 correct 50
Epoch  240  loss  0.11426507174587427 correct 50


### GPU xor 6m45s/250 epochs -> 1.62s per epoch
Epoch  0  loss  6.613845773914195 correct 35
Epoch  10  loss  4.929246212939549 correct 38
Epoch  20  loss  3.244614394280823 correct 41
Epoch  30  loss  3.4185786996348755 correct 47
Epoch  40  loss  1.308267461278373 correct 41
Epoch  50  loss  2.0170419777759 correct 45
Epoch  60  loss  3.8662104730880547 correct 47
Epoch  70  loss  1.6422476081174655 correct 48
Epoch  80  loss  1.7870460801372534 correct 48
Epoch  90  loss  1.4704598974490823 correct 48
Epoch  100  loss  1.363390468471486 correct 48
Epoch  110  loss  0.6934944748325108 correct 48
Epoch  120  loss  0.7417159084922713 correct 48
Epoch  130  loss  1.835653638306702 correct 48
Epoch  140  loss  4.000598769027761 correct 49
Epoch  150  loss  1.3499429753000727 correct 48
Epoch  160  loss  1.4383209185741497 correct 48
Epoch  170  loss  1.5347546102661485 correct 48
Epoch  180  loss  1.0794497486330654 correct 48
Epoch  190  loss  1.2437754128037484 correct 48
Epoch  200  loss  1.2149848298903625 correct 48
Epoch  210  loss  0.2968091532583476 correct 48
Epoch  220  loss  2.1193951091947105 correct 48
Epoch  230  loss  0.7776381964597234 correct 48
Epoch  240  loss  1.629146564405564 correct 48

### GPU Split 13m36s/500 epochs -> 1.632s per epoch
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