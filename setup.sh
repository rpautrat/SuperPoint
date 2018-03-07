read -p "Path of the directory where datasets are stored and read: " dir
echo "DATA_PATH = '$dir'" >> ./superpoint/settings.py

read -p "Path of the directory where experiments data (logs, checkpoints, configs) are written: " dir
echo "EXPER_PATH = '$dir'" >> ./superpoint/settings.py
