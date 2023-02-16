
.PHONY: all install build clean

ROSKO_HOME := $(PWD)

CAKE_INC_PATH  := $(CAKE_HOME)/include

INCLUDE_PATH  := $(ROSKO_HOME)/include


# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
# CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))
CFLAGS_tmp := -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L -lpthread -fopenmp
CFLAGS_tmp        += -I$(INCLUDE_PATH) -I$(CAKE_INC_PATH)
CFLAGS_tmp        += -g
CFLAGS_BLIS := $(CFLAGS_tmp)
# Add local header paths to CFLAGS

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# shared library name.
LIBROSKO      := librosko.so

ROSKO_SRC := $(ROSKO_HOME)/src
UNAME_M := $(shell uname -m) # TODO uname -m doesn't work on all systems
SRC_FILES =  $(wildcard $(ROSKO_HOME)/src/*.cpp)
# cake shared library
LIBS := -L$(CAKE_HOME) -lcake


ifeq ($(UNAME_M),aarch64)
	TARGETS = rosko_armv8
	SRC_FILES += $(ROSKO_SRC)/kernels/armv8/sparse.cpp
	CFLAGS_tmp += -O3 -mtune=cortex-a53
	LIBS += -L$(ROSKO_HOME) -lrosko_kernels
else ifeq ($(UNAME_M),x86_64)
	TARGETS = rosko_haswell
	SRC_FILES += $(ROSKO_SRC)/kernels/haswell/sparse.cpp
	CFLAGS_tmp += -mavx -mfma -mtune=haswell -O2
	LIBS += -L$(ROSKO_HOME) -lrosko_kernels
endif



CFLAGS 	:= $(filter-out -std=c99, $(CFLAGS_tmp))


# --- Primary targets ---

all: $(TARGETS) 

# 	dpcpp -fp-speculation=fast g++ $(CFLAGS) $(ROSKO_SRC)/block_sizing.cpp $(ROSKO_SRC)/rosko_sgemm.cpp \

rosko_haswell: $(wildcard *.h) $(wildcard *.c)
	g++  $(CFLAGS) $(SRC_FILES) $(LIBS) \
	-DUSE_ROSKO_HASWELL -DUSE_ROSKO_PACK -shared -o $(LIBROSKO)

rosko_armv8: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS) $(SRC_FILES) $(LIBS) \
	-DUSE_ROSKO_ARMV8 -DBLIS_ARMV8_PACK -shared -o $(LIBROSKO)

# -- Clean rules --

clean:
	rm -rf *.o *.so
