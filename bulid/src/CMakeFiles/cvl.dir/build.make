# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/Workspace/SW

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Workspace/SW/bulid

# Include any dependencies generated for this target.
include src/CMakeFiles/cvl.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cvl.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cvl.dir/flags.make

src/CMakeFiles/cvl.dir/./cvl_generated_cvl.cu.o: src/CMakeFiles/cvl.dir/cvl_generated_cvl.cu.o.depend
src/CMakeFiles/cvl.dir/./cvl_generated_cvl.cu.o: src/CMakeFiles/cvl.dir/cvl_generated_cvl.cu.o.cmake
src/CMakeFiles/cvl.dir/./cvl_generated_cvl.cu.o: ../src/cvl.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/Workspace/SW/bulid/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object src/CMakeFiles/cvl.dir//./cvl_generated_cvl.cu.o"
	cd /home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir && /usr/bin/cmake -E make_directory /home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir//.
	cd /home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir//./cvl_generated_cvl.cu.o -D generated_cubin_file:STRING=/home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir//./cvl_generated_cvl.cu.o.cubin.txt -P /home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir//cvl_generated_cvl.cu.o.cmake

# Object files for target cvl
cvl_OBJECTS =

# External object files for target cvl
cvl_EXTERNAL_OBJECTS = \
"/home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir/./cvl_generated_cvl.cu.o"

../bin/cvl: src/CMakeFiles/cvl.dir/./cvl_generated_cvl.cu.o
../bin/cvl: src/CMakeFiles/cvl.dir/build.make
../bin/cvl: /usr/local/cuda-6.5/lib/libcudart.so
../bin/cvl: /usr/lib/libopencv_vstab.so.2.4.10
../bin/cvl: /usr/lib/libopencv_tegra.so.2.4.10
../bin/cvl: /usr/lib/libopencv_imuvstab.so.2.4.10
../bin/cvl: /usr/lib/libopencv_facedetect.so.2.4.10
../bin/cvl: /usr/lib/libopencv_esm_panorama.so.2.4.10
../bin/cvl: /usr/lib/libopencv_videostab.so.2.4.10
../bin/cvl: /usr/lib/libopencv_video.so.2.4.10
../bin/cvl: /usr/lib/libopencv_ts.a
../bin/cvl: /usr/lib/libopencv_superres.so.2.4.10
../bin/cvl: /usr/lib/libopencv_stitching.so.2.4.10
../bin/cvl: /usr/lib/libopencv_photo.so.2.4.10
../bin/cvl: /usr/lib/libopencv_objdetect.so.2.4.10
../bin/cvl: /usr/lib/libopencv_ml.so.2.4.10
../bin/cvl: /usr/lib/libopencv_legacy.so.2.4.10
../bin/cvl: /usr/lib/libopencv_imgproc.so.2.4.10
../bin/cvl: /usr/lib/libopencv_highgui.so.2.4.10
../bin/cvl: /usr/lib/libopencv_gpu.so.2.4.10
../bin/cvl: /usr/lib/libopencv_flann.so.2.4.10
../bin/cvl: /usr/lib/libopencv_features2d.so.2.4.10
../bin/cvl: /usr/lib/libopencv_core.so.2.4.10
../bin/cvl: /usr/lib/libopencv_contrib.so.2.4.10
../bin/cvl: /usr/lib/libopencv_calib3d.so.2.4.10
../bin/cvl: /usr/lib/libopencv_tegra.so.2.4.10
../bin/cvl: /usr/lib/libopencv_stitching.so.2.4.10
../bin/cvl: /usr/lib/libopencv_gpu.so.2.4.10
../bin/cvl: /usr/lib/libopencv_photo.so.2.4.10
../bin/cvl: /usr/lib/libopencv_objdetect.so.2.4.10
../bin/cvl: /usr/lib/libopencv_legacy.so.2.4.10
../bin/cvl: /usr/lib/libopencv_video.so.2.4.10
../bin/cvl: /usr/lib/libopencv_ml.so.2.4.10
../bin/cvl: /usr/lib/libopencv_calib3d.so.2.4.10
../bin/cvl: /usr/lib/libopencv_features2d.so.2.4.10
../bin/cvl: /usr/lib/libopencv_highgui.so.2.4.10
../bin/cvl: /usr/lib/libopencv_imgproc.so.2.4.10
../bin/cvl: /usr/lib/libopencv_flann.so.2.4.10
../bin/cvl: /usr/lib/libopencv_core.so.2.4.10
../bin/cvl: src/CMakeFiles/cvl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../bin/cvl"
	cd /home/ubuntu/Workspace/SW/bulid/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cvl.dir/build: ../bin/cvl
.PHONY : src/CMakeFiles/cvl.dir/build

src/CMakeFiles/cvl.dir/requires:
.PHONY : src/CMakeFiles/cvl.dir/requires

src/CMakeFiles/cvl.dir/clean:
	cd /home/ubuntu/Workspace/SW/bulid/src && $(CMAKE_COMMAND) -P CMakeFiles/cvl.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cvl.dir/clean

src/CMakeFiles/cvl.dir/depend: src/CMakeFiles/cvl.dir/./cvl_generated_cvl.cu.o
	cd /home/ubuntu/Workspace/SW/bulid && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Workspace/SW /home/ubuntu/Workspace/SW/src /home/ubuntu/Workspace/SW/bulid /home/ubuntu/Workspace/SW/bulid/src /home/ubuntu/Workspace/SW/bulid/src/CMakeFiles/cvl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cvl.dir/depend

