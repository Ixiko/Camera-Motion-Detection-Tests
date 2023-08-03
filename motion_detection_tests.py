import sys
import cv2
import ctypes
import imutils
import numpy as np
from mjpeg_streamer import MjpegServer, Stream

# Import script own module(s)
from libs import sysinfo
from libs import lan
from libs import image_manipulations as iman
from sklearn.multiclass import available_if


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# start capturing
def start_capture(webcam_id, cam_settings):

  # overrides other api wishes to show settings dialog
  if sys.platform == "win32" and cam_settings['show_settings_dialog'] is True:
    cam_settings['api'] = cv2.CAP_DSHOW

  cap = cv2.VideoCapture(webcam_id , cam_settings['api'])

  # get and set camera properties
  cap_brightness_standard = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
  cap_contrast_standard   = int(cap.get(cv2.CAP_PROP_CONTRAST))
  cap_autoexposure        = int(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
  cap_mode                = int(cap.get(cv2.CAP_PROP_MODE))
  cap_gain                = int(cap.get(cv2.CAP_PROP_GAIN))
  cap_fourcc              = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
  cap_fourcc_str          = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii")
  cap_settings            = int(cap.get(cv2.CAP_PROP_SETTINGS))

  cap.set(cv2.CAP_PROP_FRAME_WIDTH , cam_settings['width'])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_settings['height'])

  if cam_settings['brightness'] is not None:
    cap.set(cv2.CAP_PROP_BRIGHTNESS  , cam_settings['brightness'])
  if cam_settings['contrast'] is not None:
    cap.set(cv2.CAP_PROP_CONTRAST    , cam_settings['contrast'])
  if cam_settings['mode'] is not None:
    cap.set(cv2.CAP_PROP_MODE        , cam_settings['mode'])

  # Open's camera settings dialog in windows
  if cam_settings['show_settings_dialog'] is True:
    cap.set(cv2.CAP_PROP_SETTINGS    , 0)

  # to check that the settings have been made
  cap_width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  cap_height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  cap_brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
  cap_contrast   = int(cap.get(cv2.CAP_PROP_CONTRAST))

  # print settings data
  print(f"Frame width and height:      w{cap_width} h{cap_height}")
  print(f"Brightness Standard/Now:     {cap_brightness_standard}/{cap_brightness}")
  print(f"Contrast Standard/Now:       {cap_contrast_standard}/{cap_contrast}")
  print("Gain:                       ", cap_gain)
  print("Capture mode:               ", cap_mode)
  print("4-character code of codec:  ", cap_fourcc_str)
  print("Automatic Exposure mode:    ", cap_autoexposure)
  print("Settings:                   ", cap_settings)

  return cap, cap_width, cap_height

def nothing(x):
 pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# calculates difference between 3 images
def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compares two images with OpenCV and returns an array of contours
def get_contours(firstImg, lastImg, compareImg, threshold=25, mode=None, method=None):
  if mode is None:
    mode =  cv2.RETR_EXTERNAL
  if method is None:
    method = cv2.CHAIN_APPROX_SIMPLE
  if mode == cv2.RETR_EXTERNAL:
    imgDiff1  = cv2.absdiff(compareImg, firstImg)
    imgDiff2  = cv2.absdiff(lastImg, compareImg)
    imgDiff   = cv2.bitwise_and(imgDiff1, imgDiff2)
    threshImg = cv2.threshold(imgDiff, threshold, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
  else:
    imgDiff   = compareImg.copy()
    threshImg = cv2.threshold(imgDiff, threshold, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
  # Find the contours and determine the corresponding OpenCV version
  contours = []
  version = cv2.__version__.split('.')[0]
  if   version == '4':
    contours, _ = cv2.findContours(threshImg.copy(), mode, method)
  elif version == '3':
    _, contours, _ = cv2.findContours(threshImg.copy(), mode, method)
  return contours, imgDiff, threshImg.copy()


# --------------------------------------------------------------------------
# WAYS TO DETECT MOTION
# --------------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# masks all colors between lower and upper colors
def mask_motion(frame, lower, upper):
  converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  imgMask   = cv2.inRange(converted, lower, upper)

  kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
  imgMask   = cv2.erode(imgMask, kernel, iterations = 1)
  imgMask   = cv2.dilate(imgMask, kernel, iterations = 1)

  imgMask   = cv2.GaussianBlur(imgMask, (3,3), 0)
  skin      = cv2.bitwise_and(frame, frame, mask = imgMask)
  return skin # np.hstack([frame, skin])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# motion-detection with OpenCV's inbuild feature to isolate the foreground
def eraseBackground_motion(motion_detector, grayImg, originalFrame):
  text = "Motion detected: none"

  # Apply the motion detector to the frame.
  motion_detected = motion_detector.apply(grayImg)

  # If motion was detected, draw a rectangle around the moving object.
  if motion_detected is not None:

    # Draw a rectangle around the moving object.
    x, y, w, h = cv2.boundingRect(motion_detected)
    cv2.rectangle(originalFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    text = f"range of motion: x{x} y{y} w{w} h{h}"

  # Draw Text
  cv2.putText(originalFrame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

  return originalFrame

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# find contours using imageDiff
def contour_motion(frame, firstFrame, lastFrame, grayImg, minArea, threshold):
  contours, imgDiff, threshImg = get_contours(firstFrame, lastFrame, grayImg, threshold) #, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contours_count = len(contours)
  frame = iman.draw_rec_contours(contours, minArea, frame)

  # Make separate copies of the images
  grayImg_with_text   = cv2.cvtColor(grayImg.copy()   , cv2.COLOR_GRAY2BGR)
  threshImg_with_text = cv2.cvtColor(threshImg.copy() , cv2.COLOR_GRAY2BGR)
  imgDiff_with_text   = cv2.cvtColor(imgDiff.copy()   , cv2.COLOR_GRAY2BGR)

  # Paste the text into the copies of the images only
  cv2.putText(grayImg_with_text  , "GRAYSCALE & GAUSSIAN IMAGE", (20, 40), cv2.FONT_HERSHEY_PLAIN, 3.2, (0, 205, 205), 3)
  cv2.putText(threshImg_with_text, "THRESHOLD IMAGE"           , (20, 40), cv2.FONT_HERSHEY_PLAIN, 3.2, (0, 205, 205), 3)
  cv2.putText(imgDiff_with_text  , "IMAGE DIFFERENCE"          , (20, 40), cv2.FONT_HERSHEY_PLAIN, 3.2, (0, 205, 205), 3)

  return frame, grayImg_with_text, threshImg_with_text, imgDiff_with_text, contours_count



# -------------------------------------------------------------------------
# MAIN PROCESS
def motion_detection_stream(video_stream_url, webcam_id=0, threshold=20):
  """Description:

  This function takes a URL of a video stream as input and
  returns a URL of an MJPEG stream as output. The function
  works by first creating an MJPEG server, then creating a
  motion detector and adding the video stream URL. The
  function then starts the video stream and reads the
  frames from the video stream. For each frame, the motion
  detection model is applied and if motion is detected, a
  rectangle is drawn around the moving object. The frame
  is then sent to the MJPEG server. The function waits for
  a key press and when the q key is pressed, the video
  stream is stopped and the MJPEG server is closed.

  Args:
    video_stream_url: The URL of the video stream.
    threshold: The threshold for detecting motion.

  Returns:
    A MJPEG stream containing the output of the motion detection function.
  """

  # need some predifinitions
  analyse_mode    = 1                                        # 1 = contour, 2 = eraseBackground, 3 = mask_motion
  minArea         = 500
  streaming       = False                                    # start mjpeg web streaming
  firstFrame      = None
  frames_to_wait  = 100                                      # frames needed for the camera warm up
  frame_nr        = 0                                        # frame counter
  im_x            = 250                                      # positioning for cv windows
  im_y            = 50                                       # positioning for cv windows
  rec_tl          = (16, 32)
  rec_br          = (200, 120)
  last_threshold  = threshold
  lower = np.array([0, 28, 40], dtype = 'uint8')             # lower color for mask_motion
  upper = np.array([20, 255, 255], dtype = 'uint8')          # upper color for mask_motion

  # your camera settings - remove the attributes to leave them unset
  cam_settings = {
      'api'                  : cv2.CAP_DSHOW,
      'width'                : 1280,
  	  'height'               : 720,
      'brightness'           : 120,
      'contrast'             : 450,
      'mode'                 : 1,
      'show_settings_dialog' : True
      }

  # Start capturing video
  cap, cam_width, cam_height = start_capture(webcam_id, cam_settings)

  # Create a MJPEG server.
  if streaming == True:
    stream = Stream("MS Lifecam HD-3000", size=(1280, 720), quality=100, fps=30)
    server = MjpegServer(video_stream_url, 8080)
    server.add_stream(stream)
    server.start()

  # Analyze mode - pre settings
  if analyse_mode == 3:
    motion_detector = cv2.createBackgroundSubtractorMOG2(varThreshold=threshold)
  elif analyse_mode == 1:
    minCC = None
    maxCC = 0
    avgCC = 0
    CCSum = 0
    CCCounts = 0

  # creates named windows and move them for a good layout
  cv2.namedWindow('Gray Image')
  cv2.namedWindow('Thresh')
  cv2.namedWindow('Image Difference')
  if streaming == False:
    cv2.namedWindow('Camera')
    cv2.resizeWindow('Camera'          , cam_width, cam_height)
    cv2.createTrackbar('threshold','Camera', 0, 100, nothing)
    cv2.setTrackbarPos('threshold', 'Camera', threshold)
    shcore   = ctypes.windll.shcore
    hresult  = shcore.SetProcessDpiAwareness(2)    # Support high DPI displays
    assert hresult == 0
    win = sysinfo.get_window_positions('Camera')
    dpi = sysinfo.get_dpi()
    tbh = 0
    if win is not None:
      title = win[0]['name']
      tbh = win[0]['size'][1] - cam_height
      sw, sh = sysinfo.get_screen_size()

      sw1 = ctypes.windll.user32.GetSystemMetrics(0)
      sh1 = ctypes.windll.user32.GetSystemMetrics(1)
      print(f"screen_w{sw1} screen_h{sh1}")
      print(f"title: {title} dpi {dpi} tbh: {tbh} camheight: {cam_height} h{im_y + cam_height + tbh}")

    scaled_w = int((cam_width+2)/3)
    scaled_h = int((cam_height+2)/3)
    cv2.resizeWindow('Gray Image'        , scaled_w, scaled_h)
    cv2.resizeWindow('Thresh'            , scaled_w, scaled_h)
    cv2.resizeWindow('Image Difference'  , scaled_w, scaled_h)

    cv2.moveWindow('Camera'            , im_x                 , im_y)
    cv2.moveWindow('Gray Image'        , im_x                 , im_y + cam_height + tbh + 7)
    cv2.moveWindow('Image Difference'  , im_x +   scaled_w + 0, im_y + cam_height + tbh + 7)
    cv2.moveWindow('Thresh'            , im_x + 2*scaled_w + 0, im_y + cam_height + tbh + 7)
  else: # Streamwindow on Monitor1 and the others on Monitor2
    cv2.moveWindow('Gray Image'        , 3845, 100)
    cv2.moveWindow('Thresh'            , 3845, 100 +   scaled_h + 1)
    cv2.moveWindow('Image Difference'  , 3845, 100 + 2*scaled_h + 2)
    print(f"get Webcam-Stream: 'http://{local_ip}:8080'")


  # Loop over the frames in the video stream.
  while True:
    # Read the next frame from the video stream.
    ret, frame = cap.read()
    # If the frame was not read successfully, break out of the loop.
    if not ret:
      break

    # cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

    frame_nr += 1
    if frame_nr == frames_to_wait:
      lastFrame = iman.to_gray_img(frame)
    if frame_nr > frames_to_wait:

      # remove colors with OpenCV
      grayImg = iman.to_gray_img(frame)

      # Saves the first received frame for comparing
      if firstFrame is None:
        firstFrame = grayImg
        continue

      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # Mode 1: gets contours of a moving object by comparing images
      if analyse_mode == 1:
        frame, imgGray, imgThresh, imgDiff, cc_count = contour_motion(frame, firstFrame, lastFrame, grayImg, minArea, threshold)
        lastFrame = grayImg
        if cc_count is not None and cc_count > 0:
          CCSum += cc_count
          CCCounts += 1
          minCC = cc_count if minCC is None else min(cc_count, minCC)
          maxCC = max(cc_count, maxCC)
          avgCC = int(CCSum/CCCounts)

        # paints a rectangle canvas element
        rect_canvas = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(rect_canvas, (rec_tl[0],rec_tl[1])   , (rec_br[0], rec_tl[1]+22), (0, 0, 255)    , thickness=cv2.FILLED)
        cv2.rectangle(rect_canvas, (rec_tl[0],rec_tl[1]+23), (rec_br[0], rec_br[1])   , (255, 255, 255), thickness=cv2.FILLED)
        frame = cv2.addWeighted(frame, 1, rect_canvas, 0.8, 0)

        cv2.rectangle(rect_canvas, (rec_tl[0],rec_tl[1])   , (rec_br[0], rec_tl[1]+22), (0, 0, 255)    , thickness=cv2.FILLED)
        cv2.rectangle(frame      , rec_tl                  , rec_br                   , (0, 0, 255)    , thickness=2)

        cv2.putText(frame , "Countour statistic", (20, 50) , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame , "min:"              , (20, 75) , cv2.FONT_HERSHEY_PLAIN  , 1.0, (0, 0, 0)      , 1)
        cv2.putText(frame , "max:"              , (20, 90) , cv2.FONT_HERSHEY_PLAIN  , 1.0, (0, 0, 0)      , 1)
        cv2.putText(frame , "avg:"              , (20, 105), cv2.FONT_HERSHEY_PLAIN  , 1.0, (0, 0, 0)      , 1)
        cv2.putText(frame , f"{minCC}"          , (70, 75) , cv2.FONT_HERSHEY_PLAIN  , 1.0, (0, 0, 0)      , 1)
        cv2.putText(frame , f"{maxCC}"          , (70, 90) , cv2.FONT_HERSHEY_PLAIN  , 1.0, (0, 0, 0)      , 1)
        cv2.putText(frame , f"{avgCC}"          , (70, 105), cv2.FONT_HERSHEY_PLAIN  , 1.0, (0, 0, 0)      , 1)


        # Display the images with text
        cv2.imshow('Gray Image'      , imutils.resize(imgGray     , scaled_w, scaled_h))
        cv2.imshow('Thresh'          , imutils.resize(imgThresh   , scaled_w, scaled_h))
        cv2.imshow('Image Difference', imutils.resize(imgDiff     , scaled_w, scaled_h))

        # get trackbar position
        tsd =  cv2.getTrackbarPos('threshold','Camera')
        if last_threshold != tsd:
          threshold = tsd
          last_threshold = tsd
          threshold = tsd
          CCSum = CCCounts = maxCC = avgCC = 0
          minCC = None



      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # Mode 2: mask skin colors
      elif analyse_mode == 2:
        frame = mask_motion(frame, lower, upper)

      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # Mode 3: Apply the motion detector to the frame.
      elif analyse_mode == 3:
        frame = eraseBackground_motion(motion_detector, grayImg, frame)
        imgGray = grayImg.copy()
        cv2.putText(imgGray  , "GRAYSCALE & GAUSSIAN IMAGE", (20, 40), cv2.FONT_HERSHEY_PLAIN, 3.2, (0, 205, 205), 3)
        cv2.imshow('Gray Image'      , imutils.resize(grayImg    , scaled_w, scaled_h))

	  # Display the original frame with the overlayed informations
    if streaming == True:
      stream.set_frame(frame)
    else:
      cv2.imshow('Camera', frame)


    # Wait for a key press. If the `q` key was pressed, break out of the loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break


  # Stop the video stream after breaking the loop.
  cap.release()

  # Close the MJPEG server.
  server.stop()

  return


# -----------------------------------------------------------
# starts main process
# -----------------------------------------------------------
if __name__ == "__main__":

  local_ip = lan.get_local_ip()
  if local_ip is None:
    local_ip = "no_ip"
  motion_detection_stream(local_ip, 0, 20)


