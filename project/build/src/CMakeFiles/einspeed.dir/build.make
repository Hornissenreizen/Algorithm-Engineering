# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build"

# Include any dependencies generated for this target.
include src/CMakeFiles/einspeed.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/einspeed.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/einspeed.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/einspeed.dir/flags.make

src/CMakeFiles/einspeed.dir/einspeed.cpp.o: src/CMakeFiles/einspeed.dir/flags.make
src/CMakeFiles/einspeed.dir/einspeed.cpp.o: /home/jonas/Documents/University/5.\ Semester/Algorithm\ Engineering/project/src/einspeed.cpp
src/CMakeFiles/einspeed.dir/einspeed.cpp.o: src/CMakeFiles/einspeed.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/einspeed.dir/einspeed.cpp.o"
	cd "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/einspeed.dir/einspeed.cpp.o -MF CMakeFiles/einspeed.dir/einspeed.cpp.o.d -o CMakeFiles/einspeed.dir/einspeed.cpp.o -c "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/src/einspeed.cpp"

src/CMakeFiles/einspeed.dir/einspeed.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/einspeed.dir/einspeed.cpp.i"
	cd "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/src/einspeed.cpp" > CMakeFiles/einspeed.dir/einspeed.cpp.i

src/CMakeFiles/einspeed.dir/einspeed.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/einspeed.dir/einspeed.cpp.s"
	cd "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/src/einspeed.cpp" -o CMakeFiles/einspeed.dir/einspeed.cpp.s

# Object files for target einspeed
einspeed_OBJECTS = \
"CMakeFiles/einspeed.dir/einspeed.cpp.o"

# External object files for target einspeed
einspeed_EXTERNAL_OBJECTS =

src/einspeed.so: src/CMakeFiles/einspeed.dir/einspeed.cpp.o
src/einspeed.so: src/CMakeFiles/einspeed.dir/build.make
src/einspeed.so: external/hptt/libhptt.a
src/einspeed.so: /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
src/einspeed.so: /usr/lib/x86_64-linux-gnu/libpthread.a
src/einspeed.so: src/CMakeFiles/einspeed.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library einspeed.so"
	cd "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/einspeed.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/einspeed.dir/build: src/einspeed.so
.PHONY : src/CMakeFiles/einspeed.dir/build

src/CMakeFiles/einspeed.dir/clean:
	cd "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src" && $(CMAKE_COMMAND) -P CMakeFiles/einspeed.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/einspeed.dir/clean

src/CMakeFiles/einspeed.dir/depend:
	cd "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project" "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/src" "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build" "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src" "/home/jonas/Documents/University/5. Semester/Algorithm Engineering/project/build/src/CMakeFiles/einspeed.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : src/CMakeFiles/einspeed.dir/depend

