.PHONY: all install build clean


INCLUDE_PATH  := $(ROSKO_HOME)/include
TEST_OBJ_PATH  := .


# Use the "framework" CFLAGS for the configuration family.
# CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))
CFLAGS_tmp := -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L 
CFLAGS_tmp        += -I$(INCLUDE_PATH) -g


# shared library name.
LIBROSKO      := librosko_kernels.so

ROSKO_SRC := $(ROSKO_HOME)/src
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),aarch64)
	KERNELS = $(ROSKO_SRC)/kernels/armv8/sparse.cpp
# 	KERNELS := $(filter-out $(ROSKO_SRC)/kernels/armv8/blis_pack_armv8.cpp, $(KERNELS)) 
	TARGETS = rosko_armv8
	CFLAGS_tmp += -O3 -mtune=cortex-a53
else ifeq ($(UNAME_M),x86_64)
	KERNELS = $(ROSKO_SRC)/kernels/haswell/sparse.cpp
# 	KERNELS := $(filter-out $(ROSKO_SRC)/kernels/haswell/blis_pack_haswell.cpp, $(KERNELS)) 
	CFLAGS_tmp += -mavx -mfma -mtune=haswell
	TARGETS = rosko_haswell
	CFLAGS_tmp += -O2
endif


CFLAGS 	:= $(filter-out -std=c99, $(CFLAGS_tmp))


# --- Primary targets ---

all: $(TARGETS) 

install:
	./install.sh
	
# 	dpcpp -fp-speculation=fast g++ $(CFLAGS) $(ROSKO_SRC)/block_sizing.cpp $(ROSKO_SRC)/rosko_sgemm.cpp \

rosko_haswell: $(wildcard *.h) $(wildcard *.c)
	g++  $(CFLAGS) $(KERNELS) \
	$(LDFLAGS) -DUSE_ROSKO_HASWELL -shared -o $(LIBROSKO)

rosko_armv8: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS) $(KERNELS) \
	$(LDFLAGS) -DUSE_ROSKO_ARMV8 -shared -o $(LIBROSKO)

# -- Clean rules --

clean:
	rm -rf *.o *.so
