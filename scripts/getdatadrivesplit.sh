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
SOURCE="chest_xray/chest_xray/test"
TARGET="chest_xray/chest_xray/val"
for file in $(ls "$SOURCE/NORMAL" | head -152)
do
mv "$SOURCE/NORMAL/$file" "$TARGET/NORMAL/"
done
for file in $(ls "$SOURCE/PNEUMONIA" | head -152)
do
mv "$SOURCE/PNEUMONIA/$file" "$TARGET/PNEUMONIA/"
done