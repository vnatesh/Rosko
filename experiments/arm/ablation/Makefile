.PHONY: all compile clean

ifneq ($(strip $(BLIS_INSTALL_PATH)),)
BLIS_INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
endif

# g++ rosko_sgemm_test.cpp -I/usr/local/include/blis -I/tmp/CAKE_on_CPU/include -L/tmp/CAKE_on_CPU  -lcake -o testing

CAKE_INC_PATH  := $(CAKE_HOME)/include
TEST_OBJ_PATH  := .

# Use the "framework" CFLAGS for the configuration family.
CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS_tmp        += -I$(CAKE_INC_PATH)

CFLAGS_blis := $(CFLAGS)
CFLAGS_blis		  += -I$(BLIS_INC_PATH)

# cake shared library
LIBCAKE      := -L$(CAKE_HOME) -lcake
CAKE_SRC := $(CAKE_HOME)/src




UNAME_P := $(shell uname -p)
SRC_FILES =  $(wildcard $(CAKE_HOME)/src/*.cpp)

ifeq ($(UNAME_P),aarch64)
	TARGETS = rosko
	TARGETS += rosko_reorder
	TARGETS += rosko_reorder_tile
	TARGETS += cake
else ifeq ($(UNAME_P),x86_64)
	TARGETS = cake_haswell
	TARGETS += cake
	TARGETS += inner
	CFLAGS_tmp		  += -g -mavx -mfma
else
	TARGETS = cake_blis
endif

CFLAGS 	:= $(filter-out -fopenmp -std=c99, $(CFLAGS_tmp))






# --- Targets/rules ------------------------------------------------------------

all: $(TARGETS)

cake_blis: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS_blis) rosko_reorder_tile.cpp $(LIBCAKE) -o rosko_sgemm_test

rosko: $(wildcard *.h) $(wildcard *.c)
	g++ -O3 $(CFLAGS) rosko.cpp $(LIBCAKE) -o rosko

rosko_reorder: $(wildcard *.h) $(wildcard *.c)
	g++ -O3 $(CFLAGS) rosko_reorder.cpp $(LIBCAKE) -o rosko_reorder

rosko_reorder_tile: $(wildcard *.h) $(wildcard *.c)
	g++ -O3 $(CFLAGS) rosko_reorder_tile.cpp $(LIBCAKE) -o rosko_reorder_tile

cake_haswell: $(wildcard *.h) $(wildcard *.c)
	dpcpp $(CFLAGS) rosko_sgemm_test.cpp $(LIBCAKE) -o rosko_sgemm_test

cake: $(wildcard *.h) $(wildcard *.c)
	g++ -O3 $(CFLAGS) cake_sgemm_test.cpp $(LIBCAKE) -o cake_sgemm_test

inner: $(wildcard *.h) $(wildcard *.c)
	g++ -O3 $(CFLAGS) inner_prod_test.cpp $(LIBCAKE) -o inner_prod_test

clean:
	rm -rf *.o *.so

