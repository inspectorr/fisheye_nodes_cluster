TODO
- clean requirements
- test_runner_x to MLBackend class
- build lib for flask service deployment
- time measurement

Workflow ideas
- providing second image in request (for bg or styling)

Nodes to create from hub
- super resolution https://tfhub.dev/captain-pool/esrgan-tf2/1

Node ideas
- portrait background replacement
- human face transformation
- real fisheye

Crazy node ideas
- recognize text then generate images on it

Other notes
- all photos must be resized to square with fields then back
- we need stepping system in MLBackend class for runners like 3

Runners current state notes
- 1 - cartoonizer. slow, ok
- 2 - image extrapolator. fast, but how to do dynamic masking?
- 3 - image styling. fast, amazing!
- 4 - low light image enhancer. slow, todo remove strange resizing
