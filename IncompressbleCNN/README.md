** Some notes on running demo **

The time stepping loop of main_driver is set up so that for a particular time step, a problem is solved coarsely up to a tolerance $e_1$ using both
the pytorch NN wrapper and the direct Stokes solver. The wrapper handles the conversion (both to-and from) of the MultiFab data to Pytorch tensor.
Both the NN-wrapped GMRES solver and direct GMRES solver are timed with this data being written to a file for later post-processing. Following the
"coarse" call, the "refine" NN-wrapped solver call is made to refine the solution, and potentially add the data to the training set as done in
the python version. 


*Information on dependices*

1. Should have a local installation of CUDA (only tested by having a local installtion of the NVIDIA CUDA Toolkit, not clear what is needed beyond nvcc).
2. Should have a local installation of CuDNN (tested with  cuDNN 8.0.4 only)
3. Build using CMAKE (Minimum version is 3.17). CMakeLists Hints should point to AMRex package, LibTorch package (only tested with libtorch 10.2), and LIBHYDRO library
4. If everything compiles, the code should run as is if a CUDA enabled device can be detected
Note: Tested using gcc,gfortran,g++ compilers only




*parameters that can easily be changed *
- batch size used for training (line 781)
- Number of epochs at every training instance (line 784)
- Number of data points collected before retraining (line 852)
- Number of data points collected before enabling mode for first time (line 853)
- Random force applied to particle (line 1548)
- x,y,z coordinate of particle (lines 1560-1562)
- number of time steps (line 1538)


*Performance data output*
- Time data for GMRES with NN-provided guesses are printed out in the file TimeData.txt
- Time data for GMRES without NN-provided guesses are printed out in the file TimeDataDirect.txt


