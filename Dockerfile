# Copyright (c) 2020-2021 Intel Corporation.
# SPDX-License-Identifier: BSD-3-Clause
FROM ubuntu:20.04 as build

RUN mkdir -p /opt/build && mkdir -p /opt/dist
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl && \
  rm -rf /var/lib/apt/lists/*

# install cmake
RUN cd /opt/build && \
    curl -LO https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh && \
    mkdir -p /opt/dist//usr/local && \
    /bin/bash cmake-3.22.1-linux-x86_64.sh --prefix=/opt/dist//usr/local --skip-license


# cleanup
RUN rm -rf /opt/dist/usr/local/include && \
    rm -rf /opt/dist/usr/local/lib/pkgconfig && \
    find /opt/dist -name "*.a" -exec rm -f {} \; || echo ""
RUN rm -rf /opt/dist/usr/local/share/doc
RUN rm -rf /opt/dist/usr/local/share/man

FROM ubuntu:20.04
COPY third-party-programs.txt /
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
  rm -rf /var/lib/apt/lists/*
# repository to install Intel(R) oneAPI Libraries
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | apt-key add -
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
  rm -rf /var/lib/apt/lists/*
# repository to install Intel(R) GPU drivers
RUN curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | apt-key add -
RUN echo "deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main" > /etc/apt/sources.list.d/intel-graphics.list

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates build-essential pkg-config gnupg libarchive13 intel-basekit-getting-started intel-oneapi-advisor intel-oneapi-ccl-devel intel-oneapi-common-licensing intel-oneapi-common-vars intel-oneapi-compiler-dpcpp-cpp intel-oneapi-dal-devel intel-oneapi-dev-utilities intel-oneapi-dnnl-devel intel-oneapi-dpcpp-debugger intel-oneapi-ipp-devel intel-oneapi-ippcp-devel intel-oneapi-libdpstd-devel intel-oneapi-mkl-devel intel-oneapi-onevpl-devel intel-oneapi-python intel-oneapi-tbb-devel intel-oneapi-vtune intel-opencl intel-level-zero-gpu level-zero level-zero-devel  && \
  rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/dist /

# install git
RUN apt-get -y update --fix-missing
RUN apt-get install -y \
  nano \
  git \
  sudo \

ENV LANG=C.UTF-8

ENV ACL_BOARD_VENDOR_PATH='/opt/Intel/OpenCLFPGA/oneAPI/Boards'
ENV ADVISOR_2022_DIR='/opt/intel/oneapi/advisor/2022.0.0'
ENV APM='/opt/intel/oneapi/advisor/2022.0.0/perfmodels'
ENV CCL_CONFIGURATION='cpu_gpu_dpcpp'
ENV CCL_ROOT='/opt/intel/oneapi/ccl/2021.5.1'
ENV CLASSPATH='/opt/intel/oneapi/mpi/2021.5.1//lib/mpi.jar:/opt/intel/oneapi/dal/2021.5.3/lib/onedal.jar'
ENV CMAKE_PREFIX_PATH='/opt/intel/oneapi/vpl/2022.0.0:/opt/intel/oneapi/tbb/2021.5.1/env/..:/opt/intel/oneapi/dal/2021.5.3:/opt/intel/oneapi/compiler/2022.0.2/linux/IntelDPCPP'
ENV CMPLR_ROOT='/opt/intel/oneapi/compiler/2022.0.2'
ENV CONDA_DEFAULT_ENV='intelpython-python3.9'
ENV CONDA_EXE='/opt/intel/oneapi/intelpython/latest/bin/conda'
ENV CONDA_PREFIX='/opt/intel/oneapi/intelpython/latest'
ENV CONDA_PROMPT_MODIFIER='(intelpython-python3.9) '
ENV CONDA_PYTHON_EXE='/opt/intel/oneapi/intelpython/latest/bin/python'
ENV CONDA_SHLVL='1'
ENV CPATH='/opt/intel/oneapi/vpl/2022.0.0/include:/opt/intel/oneapi/tbb/2021.5.1/env/../include:/opt/intel/oneapi/mpi/2021.5.1//include:/opt/intel/oneapi/mkl/2022.0.2/include:/opt/intel/oneapi/ippcp/2021.5.1/include:/opt/intel/oneapi/ipp/2021.5.2/include:/opt/intel/oneapi/dpl/2021.6.0/linux/include:/opt/intel/oneapi/dnnl/2022.0.2/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/dev-utilities/2021.5.2/include:/opt/intel/oneapi/dal/2021.5.3/include:/opt/intel/oneapi/ccl/2021.5.1/include/cpu_gpu_dpcpp'
ENV DAALROOT='/opt/intel/oneapi/dal/2021.5.3'
ENV DALROOT='/opt/intel/oneapi/dal/2021.5.3'
ENV DAL_MAJOR_BINARY='1'
ENV DAL_MINOR_BINARY='1'
ENV DNNLROOT='/opt/intel/oneapi/dnnl/2022.0.2/cpu_dpcpp_gpu_dpcpp'
ENV DPL_ROOT='/opt/intel/oneapi/dpl/2021.6.0'
ENV FI_PROVIDER_PATH='/opt/intel/oneapi/mpi/2021.5.1//libfabric/lib/prov:/usr/lib64/libfabric'
ENV FPGA_VARS_ARGS=''
ENV FPGA_VARS_DIR='/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga'
ENV GDB_INFO='/opt/intel/oneapi/debugger/2021.5.0/documentation/info/'
ENV INFOPATH='/opt/intel/oneapi/debugger/2021.5.0/gdb/intel64/lib'
ENV INTELFPGAOCLSDKROOT='/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga'
ENV INTEL_PYTHONHOME='/opt/intel/oneapi/debugger/2021.5.0/dep'
ENV IPPCP_TARGET_ARCH='intel64'
ENV IPPCRYPTOROOT='/opt/intel/oneapi/ippcp/2021.5.1'
ENV IPPROOT='/opt/intel/oneapi/ipp/2021.5.2'
ENV IPP_TARGET_ARCH='intel64'
ENV I_MPI_ROOT='/opt/intel/oneapi/mpi/2021.5.1'
ENV LD_LIBRARY_PATH='/opt/intel/oneapi/vpl/2022.0.0/lib:/opt/intel/oneapi/tbb/2021.5.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.5.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.5.1//lib/release:/opt/intel/oneapi/mpi/2021.5.1//lib:/opt/intel/oneapi/mkl/2022.0.2/lib/intel64:/opt/intel/oneapi/ippcp/2021.5.1/lib/intel64:/opt/intel/oneapi/ipp/2021.5.2/lib/intel64:/opt/intel/oneapi/dnnl/2022.0.2/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/debugger/2021.5.0/gdb/intel64/lib:/opt/intel/oneapi/debugger/2021.5.0/libipt/intel64/lib:/opt/intel/oneapi/debugger/2021.5.0/dep/lib:/opt/intel/oneapi/dal/2021.5.3/lib/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/lib:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/x64:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.5.1/lib/cpu_gpu_dpcpp'
ENV LIBRARY_PATH='/opt/intel/oneapi/vpl/2022.0.0/lib:/opt/intel/oneapi/tbb/2021.5.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.5.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.5.1//lib/release:/opt/intel/oneapi/mpi/2021.5.1//lib:/opt/intel/oneapi/mkl/2022.0.2/lib/intel64:/opt/intel/oneapi/ippcp/2021.5.1/lib/intel64:/opt/intel/oneapi/ipp/2021.5.2/lib/intel64:/opt/intel/oneapi/dnnl/2022.0.2/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/dal/2021.5.3/lib/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2022.0.2/linux/lib:/opt/intel/oneapi/ccl/2021.5.1/lib/cpu_gpu_dpcpp'
ENV MANPATH='/opt/intel/oneapi/mpi/2021.5.1/man:/opt/intel/oneapi/debugger/2021.5.0/documentation/man:/opt/intel/oneapi/compiler/2022.0.2/documentation/en/man/common::'
ENV MKLROOT='/opt/intel/oneapi/mkl/2022.0.2'
ENV NLSPATH='/opt/intel/oneapi/mkl/2022.0.2/lib/intel64/locale/%l_%t/%N:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin/locale/%l_%t/%N'
ENV OCL_ICD_FILENAMES='libintelocl_emu.so:libalteracl.so:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/x64/libintelocl.so'
ENV ONEAPI_ROOT='/opt/intel/oneapi'
ENV PATH='/opt/intel/oneapi/vtune/2022.1.0/bin64:/opt/intel/oneapi/vpl/2022.0.0/bin:/opt/intel/oneapi/mpi/2021.5.1//libfabric/bin:/opt/intel/oneapi/mpi/2021.5.1//bin:/opt/intel/oneapi/mkl/2022.0.2/bin/intel64:/opt/intel/oneapi/intelpython/latest/bin:/opt/intel/oneapi/intelpython/latest/condabin:/opt/intel/oneapi/dev-utilities/2021.5.2/bin:/opt/intel/oneapi/debugger/2021.5.0/gdb/intel64/bin:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/bin:/opt/intel/oneapi/advisor/2022.0.0/bin64:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
ENV PKG_CONFIG_PATH='/opt/intel/oneapi/vtune/2022.1.0/include/pkgconfig/lib64:/opt/intel/oneapi/vpl/2022.0.0/lib/pkgconfig:/opt/intel/oneapi/tbb/2021.5.1/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.5.1/lib/pkgconfig:/opt/intel/oneapi/mkl/2022.0.2/lib/pkgconfig:/opt/intel/oneapi/ippcp/2021.5.1/lib/pkgconfig:/opt/intel/oneapi/dpl/2021.6.0/lib/pkgconfig:/opt/intel/oneapi/dnnl/2022.0.2/cpu_dpcpp_gpu_dpcpp/../lib/pkgconfig:/opt/intel/oneapi/dal/2021.5.3/lib/pkgconfig:/opt/intel/oneapi/compiler/2022.0.2/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.5.1/lib/pkgconfig:/opt/intel/oneapi/advisor/2022.0.0/include/pkgconfig/lib64:'
ENV PYTHONPATH='/opt/intel/oneapi/advisor/2022.0.0/pythonapi'
ENV SETVARS_COMPLETED='1'
ENV TBBROOT='/opt/intel/oneapi/tbb/2021.5.1/env/..'
ENV VTUNE_PROFILER_2022_DIR='/opt/intel/oneapi/vtune/2022.1.0'
ENV _CE_CONDA=''
ENV _CE_M=''
