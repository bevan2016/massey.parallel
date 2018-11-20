#ifndef DRAW_H
#define DRAW_H

/*
  Routines for marking regions on a 2D array with initial temperature
  values. There are 3 routines included here:

  fix_boundaries0()
  fix_boundaries1()
  fix_boundaries2()

  Versions 0 and 1 will just put some temperature values along the
  edges of the plate represented by the array. Version 2 will draw a
  simple printed circuit on the array.

  You can play around with 0 and 1, but please use 2 for the final
  version of your assignment.

  All temperature values are on the Kelvin scale

 */

#include <cmath>

template<typename T>
void put_rect(Array<T, 2>& h, T value, 
	      double x1, double x2, double y1, double y2)
{
  const int npixx = h.length[1];
  const int npixy = h.length[0];

  const int ix1 = static_cast<int>(floor(x1 * npixx));
  const int ix2 = static_cast<int>(floor(x2 * npixx));
  const int iy1 = static_cast<int>(floor(y1 * npixy));
  const int iy2 = static_cast<int>(floor(y2 * npixy));

  for (int iy = iy1; iy <= iy2; ++iy)
  for (int ix = ix1; ix <= ix2; ++ix) {

    h(iy, ix) = value;

  }

}

template<typename T>
void put_circ(Array<T, 2>& h, T value, 
	      double x0, double y0, double radius)
{
  const double npixx = static_cast<double>(h.length[1]);
  const double npixy = static_cast<double>(h.length[0]);

  const double x1 = x0 - radius;
  const double x2 = x0 + radius;
  const double y1 = y0 - radius;
  const double y2 = y0 + radius;

  const int ix1 = static_cast<int>(floor(x1 * npixx));
  const int ix2 = static_cast<int>(floor(x2 * npixx));
  const int iy1 = static_cast<int>(floor(y1 * npixy));
  const int iy2 = static_cast<int>(floor(y2 * npixy));

  for (int iy = iy1; iy <= iy2; ++iy)
  for (int ix = ix1; ix <= ix2; ++ix) {
    double dx = ix / npixx - x0;
    double dy = iy / npixy - y0;
    double r = sqrt(dx * dx + dy * dy);
    if (r < radius) h(iy, ix) = value;
  }

}

template<typename T>
void connectx(Array<T, 2>& h, T value1, double x1, T value2, double x2, 
	      double y0, double wid)
{
  const double npixx = static_cast<double>(h.length[1]);
  const double npixy = static_cast<double>(h.length[0]);
  const int ix1 = static_cast<int>(floor(x1 * npixx));
  const int ix2 = static_cast<int>(floor(x2 * npixx));
  const int iy0 = static_cast<int>(floor(y0 * npixy));
  const int iwid = static_cast<int>(floor(wid * npixy));

  double diff = static_cast<double>(ix2 - ix1);

  for (int ix = ix1; ix <= ix2; ++ix) {

    double f = (ix - ix1) / diff;
    double v = value1 + f * (value2 - value1);
    for (int iy = iy0-iwid; iy <= iy0+iwid; ++iy) h(iy, ix) = v;

  }
  
}

template<typename T>
void connecty(Array<T, 2>& h, T value1, double y1, T value2, double y2, 
	      double x0, double wid)
{
  const double npixx = static_cast<double>(h.length[1]);
  const double npixy = static_cast<double>(h.length[0]);
  const int iy1 = static_cast<int>(floor(y1 * npixy));
  const int iy2 = static_cast<int>(floor(y2 * npixy));
  const int ix0 = static_cast<int>(floor(x0 * npixx));
  const int iwid = static_cast<int>(floor(wid * npixx));

  const double diff = static_cast<double>(iy2 - iy1);

  for (int iy = iy1; iy <= iy2; ++iy) {

    double f = (iy - iy1) / diff;
    double v = value1 + f * (value2 - value1);
    for (int ix = ix0-iwid; ix <= ix0+iwid; ++ix) h(iy, ix) = v;

  }
  
}

// Initialize temperatures around the edges with a steady gradient 
// from corner to corner
template<typename T>
void fix_boundaries0(Array<T, 2>& h)
{
  const double wwid = 0.005;
  const float T1 = 280;
  const float T2 = 330;
  const float T3 = 74;
  const float T4 = 290;
  const int npixy = h.length[0];
  const int npixx = h.length[1];
  for (int y = 0; y < npixy; ++y) {
    float f = (float)y / (npixy - 1);
    h(y, 0) = T1 + f * (T2 - T1);
    h(y, npixx-1) = T4 + f * (T3 - T4);
  }
  for (int x = 0; x < npixx; ++x) {
    float f = (float)x / (npixx - 1);
    h(0, x) = T1 + f * (T4 - T1);
    h(npixy-1, x) = T2 + f * (T3 - T2);
  }
  h(npixy/2, npixx/2) = 0;
}

// Initialize the plate edges with a hotspot on one of the edges
template<typename T>
void fix_boundaries1(Array<T, 2>& h)
{
  const T t1 = 2;
  const T t2 = 7;
  const int n = h.length[0];
  for (int i = 0; i < n; ++i) {
    h(0, i) = t1;
    h(n-1, i) = t2;
  }
  for (int j = 0; j < n; ++j) {
    T t = t1 + (t2 - t1) * (T)j/(T)n;
    h(j, 0) = t;
    h(j, n-1) = t;
  }

  for (int i=n/2-n/4; i<n/2+n/4; ++i) h(i, 0) = 8;
  for (int i=n/2-n/8; i<n/2+n/8; ++i) h(i, 0) = 0.3;

}

// Use this for your assignment. This initializes the image with a
// cold boundary and a "cold finger", together with some warm printed
// circuit components.
template<typename T>
void fix_boundaries2(Array<T, 2>& h)
{
  const double wwid = 0.005;
  const float T0 = 273.0;
  const float T1 = T0 + 50;
  const float T2 = T0 + 8.0;
  const float T3 = T0 + 30;
  const float T4 = T0 + 10;
  const float T5 = T0 + 5 ;
  const float T6 = T0 - 100;

  const int npixy = h.length[0];
  const int npixx = h.length[1];
  for (int y = 0; y < npixy; ++y) {
    h(y, 0) = T6;
    h(y, npixx-1) = T6;
  }
  for (int x = 0; x < npixx; ++x) {
    h(0, x) = T6;
    h(npixy-1, x) = T6;
  }

  const double x0 = 0.2;
  const double y0 = 0.7;
  const double r = 0.05;
  put_circ(h, T1, x0, y0, r);
  connectx(h, T1, x0+r, T2, 0.7, y0, wwid);
  put_rect(h, T2, 0.7, 0.76, 0.65, 0.8);

  connecty(h, T3, 0.1, T2, 0.65, 0.73, wwid);
  put_rect(h, T3, 0.6, 0.76, 0.05, 0.1);

  connecty(h, T3, 0.1, T4, 0.3, 0.62, wwid);
  connectx(h, T5, 0.2, T4, 0.62,  0.3, wwid);
  put_rect(h, T5, 0.15, 0.2, 0.2, 0.4);

  // Cold finger heat sink
  put_rect(h, T6, 0.0, 0.55, 0.5, 0.53);
}



#endif
