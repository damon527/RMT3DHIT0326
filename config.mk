# Compiler and compilation flags
cc=gcc
cxx=g++
cflags=-Wall -march=native -ansi -pedantic -O3 -fopenmp -std=c++11 -fPIC -lstdc++fs -Wno-variadic-macros -g

# MPI compiler
mpicxx=mpicxx -Wno-long-long

# LAPACK flags for dense linear algebra
lp_lflags=-llapack -lblas

# Extra link flags for app-specific dependencies (e.g. FFTW-MPI)
extra_lflags=-lfftw3_mpi -lfftw3 -lm
