running models in dev
```shell
pip install -r requirements.txt
pip install -e .
python fnc/nodes/<node name>.py
```
running microservice in dev
```shell
flask run
```

deployment
```shell
touch local_settings.py
docker compose up
```

Node ideas
- portrait background replacement
- human face transformation
- real fisheye
- recognize text then generate images on it

Links to find new models:
- https://modelzoo.co/category/computer-vision
- https://tfhub.dev/s?deployment-format=lite&subtype=module,placeholder
- https://huggingface.co/models?pipeline_tag=image-to-image&sort=downloads

Models to investigate
- gender transformation https://modelzoo.co/model/im2im
- faces to pictures https://modelzoo.co/model/domain-transfer-network
- colorization https://modelzoo.co/model/colornet
- super resolution https://tfhub.dev/captain-pool/esrgan-tf2/1
- image generation https://modelzoo.co/model/gan-cls
- image translation (zebra to horse) https://modelzoo.co/model/improved-cyclegan
- image to text https://modelzoo.co/model/im2txt

Other notes
- background can be removed with segmentation model (example: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
