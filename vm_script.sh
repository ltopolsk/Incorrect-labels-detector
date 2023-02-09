sudo apt-get update
sudo apt install pip -y
pip install -r requirements.txt
python3 xml_script.py -d './VOCdevkit/VOC2007' -f change_name -p 0.05
nohup python3 -m visdom.server &
python3 train.py train 