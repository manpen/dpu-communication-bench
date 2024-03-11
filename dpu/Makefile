BUILDDIR ?= .
NR_TASKLETS ?= 16
NR_DPUS ?= 2048

DPU_TARGET := ${BUILDDIR}/checksum_dpu

COMMON_INCLUDES := ../common
DPU_SOURCES := checksum.c

.PHONY: all clean test

COMMON_FLAGS := -Wall -Wextra -Werror -Wno-unused-function -g -I${COMMON_INCLUDES} -pg -gdwarf-4
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: ${DPU_TARGET}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) ${DPU_TARGET}


