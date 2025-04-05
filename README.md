# Sidereal visibility averaging

This package allows you to average visibilities from interferometers for similar baseline coordinates, when combining multiple observations centered on the same pointing center. 
The code is currently only written for LOFAR data but may be adjusted for other instruments as well.

Install with ```pip install sidereal-visibility-avg```

Basic example: 
```sva --msout <MS_OUTPUT_NAME> *.ms```

What the code does:
1) Make a MeasurementSet template, which includes includes all baselines, frequency, and new time resolution covering all input MS in Local Sidereal Time (LST).

2) Map baselines from input MS to output MS.
    This step makes *baseline_mapping folders with the baseline mappings in json files.

3) Interpolate new UVW data with nearest neighbours (or optionally with DP3).

4) Make new mapping between input MS and output MS, using only UVW data points.

5) Average measurement sets in the template (Stack class).
The averaging is done with a weighted average, using the FLAG and WEIGHT_SPECTRUM columns.


See de Jong et al. (2025; https://arxiv.org/pdf/2501.07374) for more details.
