pip install -q kaggle
mkdir -p ~/.kaggle
cp drive/My\ Drive/kaggle.json ~/.kaggle/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
unzip chest_xray.zip
rm chest-xray-pneumonia.zip
rm chest_xray.zip
rm -rf __MACOSX/
rm -rf sample_data/