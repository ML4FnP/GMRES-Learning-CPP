# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build

# Include any dependencies generated for this target.
include CMakeFiles/Main.ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Main.ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Main.ex.dir/flags.make

CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.o: CMakeFiles/Main.ex.dir/flags.make
CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.o: ../OfflineGMRES.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.o -c /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/OfflineGMRES.cpp

CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/OfflineGMRES.cpp > CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.i

CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/OfflineGMRES.cpp -o CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.s

# Object files for target Main.ex
Main_ex_OBJECTS = \
"CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.o"

# External object files for target Main.ex
Main_ex_EXTERNAL_OBJECTS =

Main.ex: CMakeFiles/Main.ex.dir/OfflineGMRES.cpp.o
Main.ex: CMakeFiles/Main.ex.dir/build.make
Main.ex: /home/kl748/Programs/libtorch/lib/libtorch.so
Main.ex: /home/kl748/Programs/libtorch/lib/libc10.so
Main.ex: /usr/local/cuda/lib64/stubs/libcuda.so
Main.ex: /usr/local/cuda/lib64/libnvrtc.so
Main.ex: /usr/local/cuda/lib64/libnvToolsExt.so
Main.ex: /usr/local/cuda/lib64/libcudart.so
Main.ex: /home/kl748/Programs/libtorch/lib/libc10_cuda.so
Main.ex: /home/kl748/Programs/libtorch/lib/libc10_cuda.so
Main.ex: /home/kl748/Programs/libtorch/lib/libc10.so
Main.ex: /usr/local/cuda/lib64/libcufft.so
Main.ex: /usr/local/cuda/lib64/libcurand.so
Main.ex: /usr/local/cuda/lib64/libcublas.so
Main.ex: /usr/lib/x86_64-linux-gnu/libcudnn.so
Main.ex: /usr/local/cuda/lib64/libnvToolsExt.so
Main.ex: /usr/local/cuda/lib64/libcudart.so
Main.ex: CMakeFiles/Main.ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Main.ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Main.ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Main.ex.dir/build: Main.ex

.PHONY : CMakeFiles/Main.ex.dir/build

CMakeFiles/Main.ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Main.ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Main.ex.dir/clean

CMakeFiles/Main.ex.dir/depend:
	cd /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build /home/kl748/Tutorials_Classes/pytorch_practice/torchcpp_practice/OfflineGMRES/OfflineExample/build/CMakeFiles/Main.ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Main.ex.dir/depend

