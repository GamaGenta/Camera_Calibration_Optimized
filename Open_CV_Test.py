import time

import cv2
import numpy
import ximea.xiapi

# print (cv2.getBuildInformation())
print("ximea version: " + ximea.__version__)



cam = ximea.xiapi.Camera()
#cam.open_device
cam.open_device_by("XI_OPEN_BY_SN", '39314051')

number_of_devices = cam.get_number_devices()
print("number of devices: " + str(number_of_devices)) 
#print(number_of_devices)


device_sn = cam.get_device_info_string('device_sn')
device_name = cam.get_device_info_string('device_name')
device_inst_path = cam.get_device_info_string('device_inst_path')
device_loc_path = cam.get_device_info_string('device_loc_path')
device_type = cam.get_device_info_string('device_type')
print('device info: ')
print("sn: " + str(device_sn))
print("name: " + str(device_name))
print("inst_path: " + str(device_inst_path))
print("device_loc_path: " + str(device_loc_path))
print("device_type: " + str(device_type))




downsampling = cam.get_param('downsampling')
print("downsampling: "+ str(downsampling))

exposure = cam.get_exposure()
exposure_min = cam.get_exposure_minimum()
exposure_max = cam.get_exposure_maximum()
exposure_inc = cam.get_exposure_increment()

print("exposure: "+ str(exposure))
print("exposure_min: "+ str(exposure_min))
print("exposure_max: "+ str(exposure_max))
print("exposure_inc: "+ str(exposure_inc))

cam.set_exposure(exposure)
cam.set_exposure_direct(exposure)

shutter_type = cam.get_shutter_type()
print("shutter_type: "+ str(shutter_type))


#settings
cam.set_imgdataformat('XI_RGB24')
cam.set_exposure(20000)

cam.enable_auto_wb()

#create instance of Image to store image data and metadata
img = ximea.xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

#cam.get_image(img, timeout=5000)

# get camCalibration file
calibration = numpy.load('calibration.npz')
print(calibration)

mtx = calibration['mtx']
dist = calibration['dist']

print("mtx: ")
print(mtx)
print("dist: ")
print(dist)

# undistort Image Check
#imgCheck = cv2.imread("./images/planar.jpg")
#height,width = imgCheck.shape[:2]
#camMatrixNew,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height))
#imgUndist = cv2.undistort(imgCheck, mtx, dist, None, camMatrixNew)

'''
# undistort Image
camMatrixNew,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height))
data = cv2.undistort(dataDist, mtx, dist, None, camMatrixNew) 
'''

#-------
try:
    print('Starting video. Press ESC to exit.')
    t0 = time.time()
    while True:

        
        #get data and pass them from camera to img
        cam.get_image(img)


        '''
        #create numpy array with data from camera. Dimensions of the array are 
        #determined by imgdataformat 
        data = img.get_image_data_numpy()
        '''
    
        
        # Extract data into OpenCV-compatible BGR array
        frame = img.get_image_data_numpy(invert_rgb_order=False)
        
        # Apply the undistort transformation
        undistorted_frame = cv2.undistort(frame, mtx, dist)
        
        # Display the live feed
        cv2.imshow("XIMEA Stream", frame)

        # Display the live corrected feed
        cv2.imshow("Undistorted XIMEA Stream", undistorted_frame)
        

        '''
        #show acquired image with time since the beginning of acquisition
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = '{:5.2f}'.format(time.time()-t0)
        cv2.putText(
            data, text, (900,150), font, 4, (255, 255, 255), 2
            )
        
        h, w = data.shape[:2] # Get image dimensions
        max_w, max_h = 1920, 1280 # Maximum window size
        # h, w = 1920, 1280
        scale = min(max_w / w, max_h / h) # Calculate scale factor
        # print(scale)
        win_w, win_h = int(w * scale), int(h * scale) # Resized dimensions
        # win_w, win_h = 1920, 1280

        win = 'OpenCV RGB video example'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # Resizable
        cv2.resizeWindow(win, win_w, win_h)  # Risezing window and image
        cv2.imshow(win, data)
        '''

        key = cv2.waitKey(1)
        if key == 27:
            break
        
except KeyboardInterrupt:
    cv2.destroyAllWindows()
#-------


cam.stop_acquisition()
print('Stop data acquisition...')

cam.close_device()



print('\n')
print("finish")





#cam.open_device_by("XI_OPEN_BY_SN", '39314051')
#cam.open_device_by_SN('39314051')
#device_sn = cam.get_device_info_string('device_sn')
#print("\n sn: " + str(device_sn))

#downsampling = cam.get_param('downsampling')
#print('downsampling')
#print(downsampling)

#cam.start_acquisition()
#img = ximea.xiapi.Image()
#cam.get_image(img, timeout=5000)

#cam.stop_acquisition

#cam.get_image(img)

#raw_data = img.get_img_data_raw

#image = cv2.imread(raw_data)

#cv2.imshow('Original Image', image)
#cv2.waitKey()



#for param in dir(cv2):
#    if "XI" in param:
#        print(param)




#import ximae 


#print(numpy.version)
#print(cv2.__version__)




#import ximea as xi
#cam = xi.Xi_Camera(DevID=0)
#cam.set_param('exposure',10000.0)
#img =  cam.get_image()
#cam.close()


# img = xiapi.Image()



# print(cv2)

# image = cv2.imread("img.jpg")

# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Original Image', image)
# cv2.waitKey()
# cv2.imshow('Grayscale Image', grayscale_image)