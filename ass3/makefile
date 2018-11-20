# Makefile for 159.735 Assignment 3
#
# To build just type:
#
# make heat_demo

CPP = g++

# Flags for Mac OSX
#CFLAGS = -w -O3 -arch ppc -arch i386

# FLAGS for Linux
CFLAGS = -w -O3

# Locally compiled modules
OBJS = fitsfile.o

# Link to CFITSIO libraries - modify these paths accordingly
LIBP = -LC:\Massey\159735 Parallel Programming\ass3\cfitsio
INCP = -IC:\Massey\159735 Parallel Programming\ass3\cfitsio

LIBS = -lcfitsio -lm

MODS = $(INCP) $(LIBP) $(LIBS) $(OBJS) 

BINS = heat_demo

all : $(BINS)

clean :
	rm -f $(BINS)
	rm -f *.o

# Demo program. Add more programs by making entries similar to this
heat_demo : heat_demo.cpp draw.hxx array.hxx $(OBJS)
	${CPP} $(CFLAGS) -o heat_demo heat_demo.cpp $(MODS)

# Modules compiled and linked separately
fitsfile.o : fitsfile.cpp fitsfile.h
	${CPP} $(CFLAGS) $(INCP) -c fitsfile.cpp

