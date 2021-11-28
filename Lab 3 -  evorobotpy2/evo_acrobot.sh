source ../venv/bin/activate
LANG=en_US

espy=evorobotpy2/bin/es.py
conf=evorobotpy2/xacrobot/acrobot.ini
python3 $espy -f=$conf -s=30 -g=bestgS30.npy -p
