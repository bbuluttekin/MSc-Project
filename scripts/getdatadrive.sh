pip install -q kaggle
mkdir -p ~/.kaggle
cp drive/My\ Drive/kaggle.json ~/.kaggle/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -qq chest-xray-pneumonia.zip
rm chest-xray-pneumonia.zip
rm -rf chest_xray/__MACOSX/
rm -rf sample_data/
# below two line can be activated after repo made public
# git clone https://github.com/bbuluttekin/MSc-Project.git
# mv MSc-Project/src/ .