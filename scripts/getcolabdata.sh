# This is a shell script for automaticly loading data inside colab enviroment.
# Before uploading this script its vital to add kaggle.json file to enviroment with fallowing code.

# from google.colab import files
# files.upload()

pip install -q kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
unzip chest_xray.zip
rm chest-xray-pneumonia.zip
rm chest_xray.zip
rm -rf __MACOSX/
rm -rf sample_data/