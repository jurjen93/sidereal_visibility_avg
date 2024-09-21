# Sidereal visibility averaging

This package allows you to average visibilities of interferometers over sidereal time when combining multiple observations centered on the same pointing center. 
It is currently only written for LOFAR data but may be adjusted for other instruments as well.

Install with ```pip install git+https://github.com/jurjen93/sidereal_visibility_avg```

Basic example: 
```sidereal_visibility_avg --msout <MS_OUTPUT_NAME> *.ms```

Strategy:
1) Make a template using the 'default_ms' option from ```casacore.tables``` (Template class).
       The template includes all baselines, frequency, and new time resolution covering all input MS.
       Time is converted to Local Sidereal Time (LST).

2) Map baselines from input MS to output MS.
    This step makes *baseline_mapping folders with the baseline mappings in json files.

3) Interpolate new UVW data with nearest neighbours.

4) Make new mapping between input MS and output MS, using only UVW data points.

5) Average measurement sets in the template (Stack class).
       The averaging is done with a weighted average, using the FLAG and WEIGHT_SPECTRUM columns.


The code and its pros and cons are described in de Jong et al. in prep.