from src.data.make_heatmap import GradCAM
from pathlib import Path

paths = [str(f) for f in Path("sample_data/").glob("*/*")]
model = None


def test_img_shape():
    # gcam = GradCAM(paths[0], (224, 224), model,
    #                last_convolution_layer="conv_11")
    # assert gcam.img.shape == (224, 224, 3)
    pass
