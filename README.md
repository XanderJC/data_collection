# Ardent Experiments

Hello, thank you for taking part in our experiment, we really appreciate it!

In order to run the program, please follow these steps:

This repo is pip installable - in the terminal: clone it, create a virtual env, and install it:

```shell
git clone https://github.com/XanderJC/data_collection.git

cd data_collection

pip install -r requirements.txt
pip install -e .
```
NOTE: We are using JAX which currently is not supported natively on Windows, although this can be solved by running Windows Subsystem for Linux.
This has been tested on Python 3.10.6.

Start the experiment by running
```shell
python hla/app_favourite.py
```

This will prompt you to enter your name in the terminal, do this and press enter.

Next you will be shown an example test image with descriptions of the different explanations available, please click through them to familiarise yourself with the environment.

Please then follow the rest of the instructions in app.

This shouldn't take longer than about 20mins.

Once over, a results file called results_{YOUR_NAME}.obj should have been created in the home directory. Please either email this to me (ajc340@cam.ac.uk) OR open a pull request on this repo with this file committed. 

Once again, thank you for your participation!

NOTE: If your system theme is set to 'dark mode' it may cause some visibility issues


