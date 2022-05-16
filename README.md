TODO
- clean requirements
- test_runner_x to MLBackend class
- build lib for flask service deployment
- time measurement
- all photos must be resized to square with fields then back
- we need stepping system in MLBackend class for runners like 3

Workflow ideas
- providing second image in request (for bg or styling)

Node ideas
- portrait background replacement
- human face transformation
- real fisheye

Crazy node ideas
- recognize text then generate images on it

Runners current state notes
- 1 - cartoonizer. slow, ok
- 2 - image extrapolator. fast, but how to do dynamic masking?
- 3 - image styling. fast, amazing!
- 4 - low light image enhancer. slow, todo remove strange resizing

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
- the first priority now is handle the existing models right and create api for them
- all of what is already done is image-to-image backends - this is the only abstract strategy for now
- background can be removed with segmentation model (example: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- do we really need microservices for this?...
