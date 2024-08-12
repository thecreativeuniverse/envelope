# ExoplaNet VErification and Orbital Period Estimation (ENVELOPE)

ENVELOPE is a Python module developed by I. Shrimpton under the supervision of Professor Peter Tino for a Bachelor's research project in the degree of Computer Science.
The full report is provided in [this pdf](ML_in_the_hunt_for_exoplanets.pdf).

## Abstract
The Kepler missions aimed to search for transiting exoplanets across hundreds of thousands of stars, however the data processing pipeline introduced for these missions required weeks worth of human input for manual verification.
In this paper, the ExoplaNet VErification and Orbital Period Estimation (ENVELOPE) framework is proposed to provide an alternative to NASA's Transiting Planet Search module, using a combination of data transformation and machine learning methods in order to verify the existence of a transiting planet, and learn the orbital period. 
Promising results are seen, and the orbital periods of transiting planets are found to a high degree of precision. 
With further studies, this proposed framework could provide a strong alternative to the Transiting Planet Search module.


# Installation and Running

In order to install this module, simply run the following:
```sh
git clone git@github.com:thecreativeuniverse/envelope.git
cd envelope
python -m pip install .
```

## Running options

Once you have built the module, you can import the classes `KoiLightCurve` and `FoldingLightCurve` into your own code, or alternatively run/adapt the existing [script](scripts/processing/fold_curves.py).


Run the script via:

```sh
python scripts/processing/fold_curves.py
```

Some flags can be included, detailed below:

| flag | shorthand | default | description 
|---|---|---|---|
|--debug|-d|False|Include extra print messages when running script, and generate relevant graphs throughout the process|
| --kepid | -i | None | If specified, ENVELOPE will only run on this specific kepid; otherwise, it will iterate through all kepids available through `data/init/confirmed_koi_w_params.csv` | 
| --csv | -c | `debug/output.csv` | Output CSV file for results
| --notify | -n | False | Notify you via Pushover when script is complete; requires your own `api_token` and `user_token`, to be placed in a directory called `keys`outside of the `irs084` directory |


# Module Structure & Other Info

The source code for the pip module, called `koipond` is located within the [src](src/) directory.
Scripts related to running code related to this project are located within the [scripts](scripts/) directory.

[debug.zip](debug.zip) includes some graphs generated in the final results analysis of this project.

In the early stages of this project, a deep learning approach was attempted.
However, this was not deemed successful and the new methodology as seen in the project report was soon devised. 
The code from this deep learning attempt is still available in the [Archive](ARCHIVE_DL_Attempt/) directory.

