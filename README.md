# Motion-Detection-Tests

Python script(s) to test OpenCV capabilities for motion detection

<br><br>

### to try this creepy little script you need the following libraries:

<br>

#### Operating system dependent libraries

- **OpenCV 4.6+** from here [Releases - OpenCV](https://opencv.org/releases/) - choose your platform. This script was written on Windows. I can't remember what the name of the python wrapper libraries was. If you can't remember or never knew, here is the pip command line and the name:

  ```python
  pip install OpenCV-Python
  ```

- 2 other important libraries are **numpy** and **imutils** these can be obtained with *pip* using: 

  ```cmd
  pip install numpy
  pip install imutils
  ```

- I use the library **mjpeg_streamer** to have a simple web server to stream the output:  

  ```cmd
  pip install mjpeg_streamer
  ```

- **Don't forget** to update your libraries from time to time by adding the **--upgrade** attribute to them

  ```cmd
  pip install what_I_already_installed_to --upgrade
  ```

<br>

#### Operating system dependent libraries

- Windows system

  ```cmd
  pip install pywin32
  pip install pygetwindow  
  ```

-  MacOS system

  ```terminal
  pip install pyobjc-framework-Quartz 
  ```

-  Linux system

  ```bash
  pip install python-xlib
  ```


This is really a lot of code for a small amount of features. <br>



### how to see the 3 motion detecting algorithms (ways)

- you only need to change this variable to 1, 2 or 3 in main script:

```python
analyse_mode    = 1          # 1 = contour, 2 = eraseBackground, 3 = mask_motion
```



### the most fundamental things I have found and noticed:

- To use the maximum video size of your camera you have to tell OpenCV which backend to use.  The VideoCaptureApi can be set on windows to one of these: 

	- [CAP_DSHOW](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gga023786be1ee68a9105bf2e48c700294dab6ac3effa04f41ed5470375c85a23504) = 700

	- [CAP_MSMF](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gga023786be1ee68a9105bf2e48c700294da278d5ad4907c9c0fe6d1c6104b746019) = 1400

   ```python
   cv2.CaptureVideo(0, cv2.CAP_DSHOW)
   ```
------

- and this is how you set the width and height of the video 

    ```python
     cap.set(cv2.CAP_PROP_FRAME_WIDTH , cam_width)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    ```

------

- A captured image should be converted to a gray image as it is more effective for calculations than leaving it in colour.

  ```python
  grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ```
------

- Blurring images allows for better contour detection (is that so?)

  ```python
  grayImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
  ```

<br>

### These things needs a solution:

- The image noise of the camera is often recognised as movement. Which is the most effective way to reduce noise? Is it ultimately more effective to take better hardware?

- Motion detection becomes unreliable when the camera is in automatic mode for brightness, contrast, saturation or exposure.  **Solution **found on Windows: by using `cv2.CAP_DSHOW` and set the camera property `cap.set(cv2.CAP_PROP_SETTINGS, 0)` , windows will open a dialog window. 

  

------

<br>

### Some questions are left:

- when using the erase Background feature of OpenCV, a 'ghost image' is always calculated, Even if nothing has changed in the scene.