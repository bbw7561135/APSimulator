#
# Usage:
# make <argument 1> <argument 2> ...
#
# Arguments:
# <none>:  Compiles with no compiler flags.
# debug:   Compiles with flags useful for debugging.
# fast:    Compiles with flags for high performance.
# clean:   Deletes auxiliary files.
# help:    Displays this help text.
#
# To compile with additional flags, add the argument
# EXTRA_FLAGS="<flags>"

# Define variables
PYTHON_VERSION = 3.6
ANACONDA_PATH = /Users/larsfrog/miniconda3
GCC_LIBRARTY_PATH = /usr/local/Cellar/gcc/8.2.0/lib/gcc/8
OMP_INCLUDE_PATH = ${GCC_LIBRARTY_PATH}/gcc/x86_64-apple-darwin18.2.0/8.2.0/include
COMPILER = g++-8

LIBRARY = procedural_noise.so
OBJECT_FILES = open-simplex-noise.o procedural_noise.o
COMPILATION_FLAGS = -fpic -arch x86_64 -fopenmp
LINKING_FLAGS = -shared -arch x86_64 -undefined dynamic_lookup
DEBUGGING_FLAGS = -Og -W -Wall -fno-common -Wcast-align -Wredundant-decls -Wbad-function-cast -Wwrite-strings -Waggregate-return -Wstrict-prototypes -Wmissing-prototypes -Wextra -Wconversion -pedantic -fbounds-check
PERFORMANCE_FLAGS = -O3 -ffast-math
HEADER_PATH_FLAGS = -I${ANACONDA_PATH}/include -I${ANACONDA_PATH}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include/numpy -I${ANACONDA_PATH}/include/python${PYTHON_VERSION}m -I${OMP_INCLUDE_PATH}
LIBRARY_LINKING_FLAGS = -fopenmp
LIBRARY_PATH_FLAGS = -L${GCC_LIBRARTY_PATH} -L${ANACONDA_PATH}/lib

# Make sure certain rules are not activated by the presence of files
.PHONY: all debug fast profile set_debug_flags set_fast_flags clean help

# Define default target group
all: $(LIBRARY)

# Define optional target groups
debug: set_debug_flags $(LIBRARY)
fast: set_fast_flags $(LIBRARY)

# Defines appropriate compiler flags for debugging
set_debug_flags:
	$(eval COMPILATION_FLAGS = $(COMPILATION_FLAGS) $(DEBUGGING_FLAGS))

# Defines appropriate compiler flags for high performance
set_fast_flags:
	$(eval COMPILATION_FLAGS = $(COMPILATION_FLAGS) $(PERFORMANCE_FLAGS))

# Rule for linking object files
$(LIBRARY): $(OBJECT_FILES)
	$(COMPILER) $(EXTRA_FLAGS) $(LINKING_FLAGS) $(OBJECT_FILES) $(LIBRARY_PATH_FLAGS) $(LIBRARY_LINKING_FLAGS) -o $(LIBRARY)

# Rule for compiling open-simplex-noise.c
open-simplex-noise.o: open-simplex-noise.h open-simplex-noise.c
	$(COMPILER) -c $(EXTRA_FLAGS) $(COMPILATION_FLAGS) $(HEADER_PATH_FLAGS) open-simplex-noise.c

# Rule for compiling procedural_noise.c
procedural_noise.o: procedural_noise.c
	$(COMPILER) -c $(EXTRA_FLAGS) $(COMPILATION_FLAGS) $(HEADER_PATH_FLAGS) procedural_noise.c

# Action for removing all auxiliary files
clean:
	rm -f $(OBJECT_FILES)

# Action for printing help text
help:
	@echo \nclean:   Deletes auxiliary files.\nhelp:    Displays this help text.\n\nTo compile with additional flags, add the argument\nEXTRA_FLAGS=\"<flags>\""
