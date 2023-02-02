#!/usr/bin/env python
import json
import numpy as np
import rospy
from collections import deque
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32
import PIL

from ur3e_openai.image_classification.frame_classifier import FrameClassifier

class PipeClassifier():
    def __init__(self):
        prefix = rospy.get_name()
        # How many batches are needed to cover the pipe surface
        self.batch_qty = rospy.get_param(prefix + "/batch_qty", default=5)
        # How many frames per batch
        self.batch_size = rospy.get_param(prefix + "/batch_size", default=5)
        # Bbox for cropping the images
        bbox = rospy.get_param(prefix + "/bounding_box", default=None)
        self.bbox = json.loads(bbox) if bbox is not None else None 
        
        # Load trained model (either for predict_glue or to predict_smooth)
        glue_prediction_model = rospy.get_param(prefix + "/glue_model")
        smooth_prediction_model = rospy.get_param(prefix + "/smooth_model")
        self.prediction_stage = rospy.get_param(prefix + "/stage", default="glue")
        self.glue_model = FrameClassifier(model_path=glue_prediction_model)
        self.smooth_model = FrameClassifier(model_path=smooth_prediction_model)

        # Determine the label of interest
        self.tolerance = rospy.get_param(prefix + "/tolerance", default=0.6)

        image_topic = "/camera/color/image_raw"
        publisher_topic = prefix + "/result"

        self.pub = rospy.Publisher(publisher_topic, Float32, queue_size=10)
        self.sub = rospy.Subscriber(image_topic, numpy_msg(Image), self.image_callback)

        self.image_queue = deque(maxlen=self.batch_size)
        self.batch_queue = deque(maxlen=self.batch_qty)

        self.debug = rospy.get_param(prefix + "/debug", default=False)
        if self.debug:
            cropped_image_topic = "/camera/color/image_cropped"
            self.pub_cropped_img = rospy.Publisher(cropped_image_topic, Image, queue_size=10)

        self.counter = np.arange(self.batch_qty, dtype='i') + 1

    def image_callback(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img = PIL.Image.fromarray(np.uint8(image))

        # crop image if requested
        if self.bbox is not None:
            # expected format bbox [min x, min y, max x, max y]
            img = img.crop(tuple(self.bbox))

        if self.debug:
            img = img.convert('RGB')
            cropped_msg = Image()
            cropped_msg.header.stamp = rospy.Time.now()
            cropped_msg.height = img.height
            cropped_msg.width = img.width
            cropped_msg.encoding = "rgb8"
            cropped_msg.is_bigendian = False
            cropped_msg.step = 3 * img.width
            cropped_msg.data = np.array(img).tobytes()
            self.pub_cropped_img.publish(cropped_msg)

        model = self.glue_model if self.prediction_stage == "glue" else self.smooth_model

        # One batch
        if len(self.image_queue) == self.image_queue.maxlen:
            glue_labels = {0:"N-G", 1:"G"}
            smooth_labels = {0:"N-SM", 1:"SM"}
            labels = glue_labels if self.prediction_stage == "glue" else smooth_labels

            batch_avg = np.average(np.array(self.image_queue)) 
            self.batch_queue.append(1.0 if batch_avg > self.tolerance else 0.0)

            # average predictions, assumed that target label is always 1.0
            pipe_prediction = np.sum(self.batch_queue)/float(self.batch_qty)
            print("batch predictions ", 
                 [ str(c)+'-'+labels.get(pred) for pred,c in zip(self.batch_queue, self.counter[:len(self.batch_queue)])], 
                 " task ", pipe_prediction,"%")
            
            # reset batch
            self.image_queue = deque(maxlen=self.batch_size)

            # Publish
            self.pub.publish(Float32(data=pipe_prediction))

            self.prediction_stage = rospy.get_param("/pipe_classifier/stage", default="glue")
            
            if len(self.batch_queue) == self.batch_queue.maxlen:
                self.counter = np.roll(self.counter, -1)
        
        if len(self.image_queue) < self.image_queue.maxlen:
            image = img.resize((224, 224), PIL.Image.NEAREST)
            self.image_queue.append(model.predict([image])[0])


def main():
    """ Main function to be run. """
    rospy.init_node("pipe_classifier_node")
    PipeClassifier()
    rospy.spin()

if __name__ == "__main__":
    main()
