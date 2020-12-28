# *******************************************************************

import argparse
import sys
import os

from utils import *

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
parser.add_argument('--temp-dir', type=str, default='temp/',
                    help='path to the temp directory')
parser.add_argument('--imperfection', type=str, default='none',
                    help='the type of image imperfection')
args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('[i] Imperfection present in input: ', args.imperfection)
print('###########################################################\n')

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# check temp directory
if not os.path.exists(args.temp_dir):
    print('==> Creating the {} directory...'.format(args.temp_dir))
    os.makedirs(args.temp_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.temp_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def power_law_transform(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    image = cv2.imread(image)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def median_blur(image, radius):
    image = cv2.imread(image)
    deblurred_img = cv2.medianBlur(image, radius)

    return deblurred_img


# for simple blur
def deblur_kernel(image, kernel):
    image = cv2.imread(image)
    sharpen = cv2.filter2D(image, -1, kernel)

    return sharpen


def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # Generate a matrix of motion blur kernels at any angle, the greater the degree, the higher the degree of blur
    m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function"""

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def process_img(frame):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    print('[i] ==> # detected faces: {}'.format(len(faces)))
    print('#' * 60)

    # initialize the set of information we'll displaying on the frame
    info = [
        ('number of faces detected', '{}'.format(len(faces)))
    ]

    for (i, (txt, val)) in enumerate(info):
        text = '{}: {}'.format(txt, val)
        cv2.putText(frame, text, (10, (i * 20) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    return frame


def process_defective():
    calibrated_image = cv2.imread("./temp/temp.jpg")
    img_output = process_img(calibrated_image)

    output_file = args.image[:-4].rsplit('/')[-1] + '_out_img.jpg'
    cv2.imwrite(os.path.join(args.output_dir, output_file), img_output.astype(np.uint8))
    cv2.imshow("Output image", img_output.astype(np.uint8))
    cv2.waitKey(2000)


def _main():
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    if args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            sys.exit(1)

        #######################################################
        #        STEPS
        # 1. Remove imperfection (if any)
        # 2. Detect face
        # 3. Find location(s) and size(s) of bounding box(es)
        # 4. Draw same box(es) on input image at same location(s)
        # 5. Save output image

        # Checking for imperfection
        if args.imperfection == "none":
            # carry simple detection
            temp_out = cv2.imread(args.image)
            cv2.imwrite(os.path.join(args.temp_dir, "temp.jpg"), temp_out.astype(np.uint8))

            process_defective()

        elif args.imperfection == "low_light":
            # apply power law transform (gamma correction)
            temp_out = power_law_transform(args.image, 2)
            cv2.imwrite(os.path.join(args.temp_dir, "temp.jpg"), temp_out.astype(np.uint8))

            process_defective()

        elif args.imperfection == "noise":
            # apply noise removal techniques
            temp_out = median_blur(args.image, 5)
            cv2.imwrite(os.path.join(args.temp_dir, "temp.jpg"), temp_out.astype(np.uint8))

            process_defective()

        elif args.imperfection == "blur":
            # remove imperfection and filter
            # defining some kernels to deblur
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

            temp_out = deblur_kernel(args.image, sharpen_kernel)
            cv2.imwrite(os.path.join(args.temp_dir, "temp.jpg"), temp_out.astype(np.uint8))

            process_defective()

        elif args.imperfection == "motion_blur":
            # remove imperfection and filter

            # Apply Wiener Filter
            defective_image = cv2.imread(args.image)
            filtered_img = motion_blur(defective_image, degree=50, angle=-45)

            cv2.imwrite(os.path.join(args.temp_dir, "temp.jpg"), filtered_img.astype(np.uint8))

            process_defective()


    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_out_vid.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)

    # ------------------- 2nd PART -------------------

    # Get the video writer initialized to save the output video
    if args.video:
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:

            has_frame, frame = cap.read()

            # Stop the program if reached end of video
            if not has_frame:
                print('[i] ==> Done processing!!!')
                print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
                cv2.waitKey(1000)
                break

            img_output = process_img(frame)

            video_writer.write(img_output.astype(np.uint8))

            cv2.imshow(wind_name, img_output)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break

        cap.release()
        cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
