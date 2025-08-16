from skimage.data import lfw_subset
import argparse
import numpy as np
import cv2 as cv
import csv
import os
def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError
parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
if __name__ == '__main__':
    
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (100, 100),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
with os.scandir('C:\\Users\\cspde\\facial_dataset') as entries:
        faces_vector=[]
        with os.scandir('C:\\Users\\cspde\\facial_dataset\\8') as imgs:
            for img1 in imgs:
                tm = cv.TickMeter()
                img1 = cv.imread('C:\\Users\\cspde\\facial_dataset\\8\\'+img1.name)
                img1Width = 100
                img1Height = 100
                img1 = cv.resize(img1, (img1Width, img1Height))
                tm.start()
                detector.setInputSize((img1Width, img1Height))
                faces1 = detector.detect(img1)
                    
                tm.stop()
                assert faces1[1] is not None, 'Cannot find a face in {}'.format()
                # Draw results on the input image
                #visualize(img1, faces1, tm.getFPS())
                # Save results if save is true
                if args.save:
                    print('Results saved to result.jpg\n')
                    cv.imwrite('result.jpg', img1)
                # Visualize results in a new window
                #cv.imshow("image1", img1)
                tm.reset()
                tm.start()
                tm.stop()
                recognizer = cv.FaceRecognizerSF.create(
                args.face_recognition_model,"")           
                face1_align = recognizer.alignCrop(img1, faces1[1][0])
                # Extract features
                face1_feature = recognizer.feature(face1_align)
            
                faces_vector.append(face1_feature[0])
        #with open('names4.csv', 'w',newline = '') as csvfile:
                    #csvwriter = csv.writer(csvfile)
                    #csvwriter.writerows(faces_vector)
        print("faces_vector", faces_vector)
        print("COV: ", np.cov(faces_vector))
