.PHONY: all compile clean

ifneq ($(strip $(BLIS_INSTALL_PATH)),)
BLIS_INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
endif

# g++ cake_sgemm_test.cpp -I/usr/local/include/blis -I/tmp/CAKE_on_CPU/include -L/tmp/CAKE_on_CPU  -lcake -o testing

CAKE_INC_PATH  := $(CAKE_HOME)/include
TEST_OBJ_PATH  := .

# Use the "framework" CFLAGS for the configuration family.
CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS_tmp        += -I$(CAKE_INC_PATH)
CFLAGS_tmp		  += -g -mavx -mfma
CFLAGS 	:= $(filter-out -fopenmp -std=c99, $(CFLAGS_tmp))

CFLAGS_blis := $(CFLAGS)
CFLAGS_blis		  += -I$(BLIS_INC_PATH)

# cake shared library
LIBCAKE      := -L$(CAKE_HOME) -lcake
CAKE_SRC := $(CAKE_HOME)/src


# --- Targets/rules ------------------------------------------------------------

all: cake 

blis: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS_blis) cake_sgemm_test.cpp $(LIBCAKE) -o cake_sgemm_test1

cake: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS) rosko_sgemm_test.cpp ../../common/*.c $(LIBCAKE) -o rosko_sgemm_test

# dpcpp -fp-speculation=fast	g++ $(CFLAGS) cake_sgemm_test.cpp $(LIBCAKE) -o cake_sgemm_test

clean:
	rm -rf *.o *.so


