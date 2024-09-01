import tensorflow as tf
import coremltools as ct
import json

print(tf.__version__)

model_path = '/Users/seohyeong/Projects/ShroomAI/ShroomAI/ckpt/model_20240827_040406/model_weight_finetune.keras'
label_map_path = '/Users/seohyeong/Projects/ShroomAI/ShroomAI/ckpt/model_20240827_040406/label_map.json'

model = tf.keras.models.load_model(model_path)

# model.author = 'seohyeong jeong'
# model.short_description = 'Mushroom Image Classification'
# model.input_description['image'] = 'Takes as input an image of a mushroom'
# model.output_description['output'] = 'Prediction of the mushroom species'

with open(label_map_path) as f:
    label_map = json.load(f)

class_labels = list(label_map.keys())

# ref (inputs, output): https://apple.github.io/coremltools/docs-guides/source/convert-tensorflow-2-bert-transformer-models.html
# ref (classifier_config): https://apple.github.io/coremltools/docs-guides/source/classifiers.html
# ref (tf coreml version compatibility): https://github.com/tensorflow/tensorflow/issues/39001

# tf: 2.17.0 (colab), keras: 3.3.3 (colab), coreml: 7.0
# tf: 2.13.0, keras: 2.14.0, coreml: 6.3.0
image_input = ct.ImageType(name="image", shape=(1, 224, 224, 3,), bias=[-1, -1, -1], scale=1/127.5)
classifier_config = ct.ClassifierConfig(class_labels)

model = ct.converters.convert(model, source='tensorflow', convert_to='mlprogram', 
                              inputs=[image_input], classifier_config=classifier_config)

model.save('/Users/seohyeong/Projects/ShroomAI/ShroomAI/ckpt/model_weight_finetune.mlmodel')