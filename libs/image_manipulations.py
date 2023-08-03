import cv2
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Performing histogram equalization
def histeq(image, nbr_bins = 256):
  """ Histogram equalization  """
  # get the image histogram
  imhist, bins = np.histogram(image.flatten(), nbr_bins, [0, 256])
  cdf = imhist.cumsum()
  #normalization of the image
  cdf = imhist.max()*cdf/cdf.max()
  cdf_mask = np.ma.masked_equal(cdf, 0)
  cdf_mask = (cdf_mask - cdf_mask.min())*255/(cdf_mask.max()-cdf_mask.min())
  cdf = np.ma.filled(cdf_mask,0).astype('uint8')
  return cdf[image.astype('uint8')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Grayscales and blurs an image
def to_gray_img(img):
  #img = imutils.resize(img, 640, 480)
  grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return cv2.GaussianBlur(grayImg, (11, 11), 10)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Draws contours to image
def draw_contours(contours, image):
  ''' Remark!
      In order for the contours to be drawn correctly, the following
      parameters must be passed to get_contours() with these values:
         mode    = cv2.RETR_TREE
         method  = cv2.CHAIN_APPROX_NONE
  '''
  image_copy = image.copy()
  cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
  return image_copy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Draws rectangels instead of contours
def draw_rec_contours(contours, minArea, Img):
  texta = ""
  # contours = imutils.grab_contours(contours)
  contours_count = len(contours)
  text = f"moving objects: {contours_count}"
  if contours_count > 0:
    for c in contours:
      if cv2.contourArea(c) < minArea:
        continue
      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 1)

  # Draw Text
  #cv2.putText(Img, text , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # to fast to read it

  return Img

