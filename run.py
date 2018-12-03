

from tensorflow.contrib.keras import utils, models, layers
import cv2
import os
import numpy as np 
import h5py
from moviepy.editor import VideoFileClip

class DigitClassifier:
    
    def __init__(self, isVideo=False):
        # self.model = self.create_model()
        # self.model.load_weights('/model_weights/custom_1.h5')
        self.model = models.load_model('model_weights/vgg_pretrained_1.h5')
        self.prev_detections = None
        self.frames_since_full_search = 0
        self.frame_count = 0
        self.isVideo = isVideo

    def detectDigits(self, img, isVideo=False):
        # for video, if detections have been made previously, only search around those detected digit areas
        # also, once every 20 frames, do a full image search
        start_x = 0
        start_y = 0
        end_x = 600
        end_y = 600
        if self.prev_detections != None and len(self.prev_detections) > 0 and self.frames_since_full_search < 10:
            min_x = 600
            min_y = 600
            max_x = 0
            max_y = 0
            for detection in self.prev_detections:
                box = detection[2]
                min_x = min(min_x, box[0])
                min_y = min(min_y, box[1])
                max_x = max(max_x, box[2])
                max_y = max(max_y, box[3])
            start_x = max(min_x - 80, 0)
            start_y = max(min_y - 80, 0)
            end_x = min(max_x + 80, 600)
            end_y = min(max_y + 80, 600)
        else:
            self.frames_since_full_search = 0
                
        self.frames_since_full_search += 1
        count = 0
        detections = []
        for win_size in range(24, 49, 24):
            step = int(win_size / 3)
            print('-------- New Step Size: {} ---------'.format(step))
            for i in range(start_y, end_y - win_size + 1, step):
                for j in range(start_x, end_x - win_size + 1, step):
                    sub_img = img[i:i+win_size, j:j+win_size]
                    prediction, certainty = self.makePrediction(sub_img, count)
                    count += 1
                    if prediction is not None:
                        #save prediction
                        box = (j, i, j+win_size, i+win_size)
                        detection = (prediction, certainty, box)
                        detections.append(detection)
                    if count % 1000 == 0:
                        print('Count: ', count)
        
        return detections, count
                    


    def makePrediction(self, img, count):
        resized = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        # preprocess images - 0 mean, normalize pixel values
        
        norm_image = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        arr = np.array([norm_image])
        predictions = self.model.predict(arr)
        
        predictions = predictions[0]
        max_val = np.max(predictions)
        index = np.where(predictions == max_val)
        prediction = index[0][0]
        
        if max_val > .985 and prediction != 2 and prediction != 1 and prediction != 3:
            print('Prediciton {}: '.format(count), prediction)
            print('Prediciton certainty: ', max_val)
            cv2.imwrite('pipeline/{}.png'.format(count), resized)
            return prediction, max_val
        else:
            return None, None

    def nonMaxSuppresion(self, detections):
        # sort detections by highest probability
        # starting with most likely, remove any overlapping boxes (that have same detected class?)
        sort_detections = sorted(detections, key=lambda x: x[1])
        final = []
        removed = []
        for detection in sort_detections:
            box = detection[2]
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            if box in removed:
                continue
            # remove overlapping
            print('Box 1 - prediciton: {} ({}, {}) ({}, {}'.format(detection[0], box[0], box[1], box[2], box[3]))
            for detection2 in detections:
                box2 = detection2[2]
                top_left2 = (box2[0], box2[1])
                bottom_right2 = (box2[2], box2[3])
                is_right_of = top_left2[0] > bottom_right[0]
                is_left_of = bottom_right2[0] < top_left[0]
                is_below = top_left2[1] > bottom_right[1]
                is_above = bottom_right2[1] < top_left[1]
                if not (is_right_of or is_left_of or is_below or is_above):
                    # overlapping
                    print('Box 2 - prediciton: {} ({}, {}) ({}, {})'.format(detection2[0], box2[0], box2[1], box2[2], box2[3]))
                    removed.append(box2)
            final.append(detection)
        return final


    def alt_nonMaxSuppresion(self, detections):
        overlapThresh = 0.25

        boxes = [[x for x in d[2]] for d in detections]
        boxes = np.array(boxes)
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
    
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
    
        # initialize the list of picked indexes	
        pick = []
    
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
    
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
    
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
    
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
    
        # return only the bounding boxes that were picked
        # print(pick)
        result = []
        for idx in pick:
            result.append(detections[idx])
        return result

    def removeConcentricRectangles(self, detections):

        result = []
        for idx, detection in enumerate(detections):
            box = detection[2]
            contained = False
            for idx2, d in enumerate(detections):
                if idx == idx2:
                    continue
                box2 = d[2]
                topLeft = box[0] >= box2[0]-10 and box[1] >= box2[1]-10
                bottom_right = box[2] <= box2[2]+10 and box[3] <= box2[3]+10
                if topLeft and bottom_right:
                    contained = True
                    break
            if not contained:
                result.append(detection)
        return result

    def drawBoxes(self, detections, img, textLeftAligned=False):
        result = img.copy()
        for detection in detections:
            box = detection[2]
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (255,0,0), thickness=4)
            prediction = detection[0]
            if prediction > 2:
                prediction -= 1
            if textLeftAligned:
                cv2.putText(result, '{}'.format(prediction), (box[0]-20, box[1]+15), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))
            else:
                cv2.putText(result, '{}'.format(prediction), (box[0]+15, box[1]-5), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))
        return result

    def processImage(self, img):
        if not self.isVideo or self.frame_count % 5 == 0:
            blurred = cv2.bilateralFilter(img,9,75,75)
            detections, win_count = self.detectDigits(blurred)
            classes_detected = set([d[0] for d in detections])
            suppressed_detections = []
            for c in classes_detected:
                single_class_detections = [d for d in detections if d[0] == c]
                single_class_detections_supressed = self.alt_nonMaxSuppresion(single_class_detections)
                suppressed_detections.extend(single_class_detections_supressed)
            suppressed_detections = self.removeConcentricRectangles(suppressed_detections)
            self.prev_detections = suppressed_detections
        else:
            suppressed_detections = self.prev_detections

        result = self.drawBoxes(suppressed_detections, img)
        self.frame_count += 1

        cv2.imwrite('box_images/output.png', result)
        print('Final window count: ', win_count)
        # classifier.makePrediction(img, 0)
        return result

if __name__ == "__main__":
    img = cv2.imread('project_images/final2_resized.png')
    # resized_img = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('project_images/final5_resized.png', resized_img)

    classifier = DigitClassifier(isVideo=False)
    classifier.processImage(img)


    # input_video = 'project_images/1.mp4'
    # output_video = 'project_images/1_processed.mp4'
    # clip = VideoFileClip(input_video)
    # classifier = DigitClassifier()
    # result = clip.fl_image(classifier.process_image)
    # result.write_video_file(output_video, audio=False)


