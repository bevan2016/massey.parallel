/******************************************************************************

Start up demo program for 159735 Assignment 3 Semester 1 2013

All this does is initialize the image and write it to a file.

To compile:

make heat_demo

To run (for example to make a 100X100 pixel image):

./heat_demo 100


******************************************************************************/
#include <iostream>

#include "arrayff.hxx"
#include "draw.hxx"

int main(int argc, char* argv[]) 
{
  // X and Y dimensions. Force it to be a square.
  const int npix = atoi(argv[1]);
  const int npixx = npix;
  const int npixy = npix;
  const int ntotal = npixx * npixy;

  // Images as 2D arrays: h is the current image, g is the updated
  // image. To access individual pixel elements, use the () operator. 
  // Note: that y is the first index (to reflect row major
  // order). Eg: h(y, x) = fubar
  Array<float, 2> h(npixy, npixx), g(npixy, npixx);

  // Draw the printed circuit components
  fix_boundaries2<float>(h);

  // This is the initial value image where the boundaries and printed
  // circuit components have been fixed
  dump_array<float, 2>(h, "plate0.fit");

  // Complete the sequential version to compute the heat transfer,
  // then make a parallel version of it
}
