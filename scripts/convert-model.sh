rm -rf static/group1*
rm static/model.json
tensorflowjs_converter --input_format keras \
                       models/final_model.h5 \
                       static/