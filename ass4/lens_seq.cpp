/* 
   159735 Parallel Programming

   Startup program for sequential implementation of simulation by ray
   tracing of gravitational lensing.
 */
#include <ctime>

#include <iostream>
#include <string>

#include <cmath>

#include "lenses.h"
#include "arrayff.hxx"

// Global variables! Not nice style, but we'll get away with it here.

// Boundaries in physical units on the lens plane
const float WL  = 2.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

// Used to time code. OK for single threaded programs but not for
// multithreaded programs. See other demos for hints at timing CUDA
// code.
double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks = clock1 - clock2;
  double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
  return diffms; // Time difference in milliseconds
}

int main(int argc, char* argv[]) 
{
  // Set up lensing system configuration - call example_1, _2, _3 or
  // _n as you wish. The positions and mass fractions of the lenses
  // are stored in these arrays
  float* xlens;
  float* ylens;
  float* eps;
  const int nlenses = set_example_n(100, &xlens, &ylens, &eps);
  std::cout << "# Simulating " << nlenses << " lens system" << std::endl;

  // Source star parameters. You can adjust these if you like - it is
  // interesting to look at the different lens images that result
  const float rsrc = 0.1;      // radius
  const float ldc  = 0.5;      // limb darkening coefficient
  const float xsrc = 0.0;      // x and y centre on the map
  const float ysrc = 0.0;

  // Pixel size in physical units of the lens image. You can try finer
  // lens scale which will result in larger images (and take more
  // time).
  const float lens_scale = 0.0025;

  // Size of the lens image
  const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
  const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
  std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;

  // Put the lens image in this array
  Array<float, 2> lensim(npixy, npixx);

  clock_t tstart = clock();

  // Draw the lensing image map here. For each pixel, shoot a ray back
  // to the source plane, then test whether or or not it hits the
  // source star
  const float rsrc2 = rsrc * rsrc;
  float xl, yl, xs, ys, sep2, mu;
  float xd, yd;
  int numuse = 0;
  for (int iy = 0; iy < npixy; ++iy) 
  for (int ix = 0; ix < npixx; ++ix) { 
    // need position on lens in physical units
    yl = YL1 + iy * lens_scale;
    xl = XL1 + ix * lens_scale;

    // shoot a ray back to the source plane - make the appropriate
    // call to shoot() in lenses.h
    shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);

    // does the ray hit the source star?
    xd = xs - xsrc;
    yd = ys - ysrc;
    sep2 = xd * xd + yd * yd;
    if (sep2 < rsrc2) {
      mu = sqrt(1 - sep2 / rsrc2);
      lensim(iy, ix) = 1.0 - ldc * (1 - mu);
    }
  }

  clock_t tend = clock();
  double tms = diffclock(tend, tstart);
  std::cout << "# Time elapsed: " << tms << " ms " << numuse << std::endl;

  // Write the lens image to a FITS formatted file. You can view this
  // image file using ds9
  dump_array<float, 2>(lensim, "../lens_s100.fit");

  delete[] xlens;
  delete[] ylens;
  delete[] eps;
}

