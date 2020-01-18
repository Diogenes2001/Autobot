# Autobot
Machine learning model and accompanying user interface that predicts auto insurance payouts

This was built as a proof of concept for the Intact Datathon 2019, so currently the UI does not connect to the model. However, I plan to connect them in the future.

To run the model:

Unzip the files in the `data` folder and place them in the main directory. Make sure you have pip installed, then run

`pip install tensorflow`

in the terminal. To see the model predict the 3 types of auto insurance payouts, run

`py collision.py`

`py comprehensive.py`

`py dcpd.py`

To run the UI:
Run the `autobot` executable.
