
# Description
This project provide a **single** tensorflow model implemented the mtcnn face detector.
 It is very handy for face detection in python and easy for deployment with tensorflow.
 The model is converted and modified from the original author's caffe model.
 
 For more detail about mtcnn, see the
  [original project](https://github.com/kpzhang93/MTCNN_face_detection_alignment).

# Requirement
- tensorflow >= 2
- opencv python binding (for reading image and show the result)

# Run
```bash
# for tensorflow 2.0
python mtcnn_tfv2.py test_image.jpg

# A demo shows how to use tensorflow dataset api
# to accelerate detection with multi-cores. This is
# especially useful for processing large amount of
# small image data in a powerful server.
python mtcnn_data.py imglist.txt result
```
# Input and Ouput
## Input: 
 BGR image.
## Output:
- box: bouding box, 2D float tensor with format [[y1, x1, y2, x2], ...]
- prob: confidence, 1D float tensor with format [x, ...]
- landmarks: face landmarks, 2D float tensor with format[[y1, y2, y3, y4, y5, x1, x2, x3, x4, x5], ...]
