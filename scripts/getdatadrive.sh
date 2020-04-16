pip install -q kaggle
mkdir -p ~/.kaggle
cp drive/My\ Drive/kaggle.json ~/.kaggle/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -qq chest-xray-pneumonia.zip
#unzip -qq chest_xray.zip
rm chest-xray-pneumonia.zip
#rm chest_xray.zip
rm -rf chest_xray/__MACOSX/
rm -rf sample_data/