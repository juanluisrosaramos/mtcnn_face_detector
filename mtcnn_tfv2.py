import argparse
#import tkinter
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def mtcnn_fun(img, min_size, factor, thresholds):
    with open('./mtcnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    with tf.device('/cpu:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    print(box, prob, landmarks)
    return box, prob, landmarks

# wrap graph function as a callable function
mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[3], dtype=tf.float32)
])

def main(args):
    img = cv2.imread(args.image)

    bbox, scores, landmarks = mtcnn_fun(img, 40, 0.7, [0.6, 0.7, 0.8])
    bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()

    print('total box:', len(bbox))
    for box, pts in zip(bbox, landmarks):
        box = box.astype('int32')
        #Show face
        img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
        pts = pts.astype('int32')        
        #Show landmarks
        
        #2D float tensor with format[[y1, y2, y3, y4, y5, x1, x2, x3, x4, x5], ...]
        #left eye, right eye, nose, left mouth, right mouth
        for i in range(2):
            img = cv2.circle(img, (pts[i+5], pts[i]), 1, (0, 255, 0), 2)          
    eyesCenter = ((pts[5] + pts[6]) // 2,(pts[0] + pts[1]) // 2)
    
    print('eyesCenter',eyesCenter)
    img = cv2.circle(img, eyesCenter, 1, (255, 0, 0), 2)
    imgplot = plt.imshow(img)
    plt.show()
    #cv2.imshow('image', img)
    #cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tensorflow mtcnn')
    parser.add_argument('image', help='image path')
    args = parser.parse_args()
    main(args)