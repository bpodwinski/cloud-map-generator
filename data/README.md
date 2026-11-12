# Reference data

## `earth_cloud_fraction_bluemarble_2048.png`

Earth's cloud fraction, 2048x1024, 8-bit grayscale, north-up. It is the reference every map is
measured against (`measure.py`), and the target of `gen.py --match-earth`.

- **Source:** NASA Visible Earth, *Blue Marble* (2001, Terra MODIS: land and ocean from June to
  September 2001, the clouds a composite of three days, the poles from infrared), as mirrored by NOAA
  Science On a Sphere (<https://sos.noaa.gov/catalog/datasets/blue-marble/>). The clouded and the
  cloud-free 4096 images.
- **Credit:** NASA Goddard Space Flight Center / NASA Visible Earth. NASA imagery is public domain;
  the credit is requested.
- **Processing:** the clouded composite inverted against the cloud-free image (divisor floored at 0.4,
  a 0.03 deadband against JPEG noise), then box-averaged to 2048x1024. Sea ice and the Antarctic
  sheet invert as cloud: the polar caps (past 70 degrees) are contaminated, and the target profile
  excludes them.
