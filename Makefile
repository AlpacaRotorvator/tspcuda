TARGET = tsp 
LIBS = -lm -lcurand
CC = gcc
CFLAGS = -g -I. -Wall

NVCC := nvcc -ccbin $(CC)
NVCCFLAGS := -m64

ALL_CFLAGS := $(NVCCFLAGS)
ALL_CFLAGS += $(addprefix -Xcompiler ,$(CFLAGS))

ALL_LDFLAGS := $(ALL_CFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES  := -I /opt/cuda/samples/common/inc

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.c, %.o, $(wildcard *.c))
HEADERS = $(wildcard *.h)

%.o: %.c $(HEADERS)
		$(NVCC) $(INCLUDES) $(ALL_CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
		$(NVCC) $(ALL_LDFLAGS) $(OBJECTS) $(LIBS) -o $@

clean: 
		-rm -f *.o
		-rm -f $(TARGET)
		-rm -f minpath.dot
