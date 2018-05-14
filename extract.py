
import base64
import json
import os
import subprocess
import cv2
import numpy as np
# Kasuta python 2.7.10
def png2image(png):
      return(cv2.imdecode(np.fromstring(png,np.uint8),cv2.IMREAD_COLOR))

def image_add_timestamp(image,timestamp):
  fonts=[cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_ITALIC]
  f = fonts[1]
  cv2.putText(image,str(timestamp), (10,460), f, 2, (128,0,0),2)
  return(image)

def image_add_text(image,text):
  fonts=[cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_ITALIC]
  f = fonts[1]
  cv2.putText(image,str(text), (10,460), f, 1, (255,0,0),2)
  return(image)

def extract_image_with_corresponding_values(image, coords, coords3D, const2Dsize, const3Dvalue):
    # Conversion on how many units in 3d is 1 pixel in 2d
    conversion = const3Dvalue / const2Dsize
    # Calculate new sizes
    newWidth = extractConvertedPixelSize(abs(coords3D[3]-coords3D[0]), conversion)
    newHeight = extractConvertedPixelSize(abs(coords3D[4]-coords3D[1]), conversion)
    if newWidth != 0 and newHeight != 0:
        # Crop the image so it could be easier to resize
        croppedImage = image[coords[1]:coords[3], coords[0]: coords[2]]
        # Resize the image
        resizedImage = cv2.resize(croppedImage, (newWidth, newHeight), interpolation = cv2.INTER_AREA)
        # Create a new image and position croppedImage in the middle
        blank_image = np.zeros((const2Dsize, const2Dsize, 3), np.uint8)
        height, width, channel = resizedImage.shape
        if height is not None and width is not None:
            paddingX = int((const2Dsize - width)/2)
            paddingY = int((const2Dsize - height)/2)
            if paddingX > 0 and paddingY > 0:
                for y in range(0, height):
                    for x in range(0, width):
                        blank_image[paddingY+y, paddingX+x] = resizedImage[y, x]
                return blank_image
    return None


def extractConvertedPixelSize(diff3D, conversion):
    return int(round((diff3D/conversion), 0))

#Muudame piksli värvi ümber, et eristada
def putPixelsForTest(image, whitePixLocations):
    for coord in whitePixLocations:
        image[coord[0], coord[1]] = [0,0,255]
    return(image)

#Filtreerib välja väärtused, mis pole bounding box'i sees
def filterOutOuterValues(coordList, coords):
    whitePixelsInFrame = []
    for coord in coordList:
        # Ei saa üldse aru kas bb väärtused on vastupidised või numpy hoiab väärtuseid nagu ( Ycoord, Xcoord )
        if coord[1] > min(coords[0], coords[2]) and coord[1] < max(coords[0],coords[2]) and coord[0] > min(coords[1], coords[3]) and coord[0] < max(coords[1], coords[3]):
            whitePixelsInFrame.append(coord)
    return whitePixelsInFrame

#Filtreerib ja tagastab mediaan väärtuse
def filterOutOuterValuesAndReturnCenter(coordList, coords):
    arrX = []
    arrY = []
    for coord in coordList:
        if (coord[1] > min(coords[0], coords[2])) and (coord[1] < max(coords[0],coords[2])) and (coord[0] > min(coords[1], coords[3])) and (coord[0] < max(coords[1], coords[3])):
            arrX.append(coord[1])
            arrY.append(coord[0])
    if len(arrX) < 1 or len(arrY) < 1:
        return None
    return (int(np.median(np.sort(arrX))), int(np.median(np.sort(arrY))))

#Joonistab kasti antud koordinaatidega
def draw_center_box_2D(image, coords):
    cv2.line(image, (coords[0], coords[1]), (coords[2], coords[1]), (0,0,255), 1)
    cv2.line(image, (coords[0], coords[3]), (coords[2], coords[3]), (0,0,255), 1)
    cv2.line(image, (coords[0], coords[1]), (coords[0], coords[3]), (0,0,255), 1)
    cv2.line(image, (coords[2], coords[1]), (coords[2], coords[3]), (0,0,255), 1)
    return (image)

#Leiab mediaani
def findMedianValue(whitePixLocations):
    arrX = [(lambda i: i[1]) (i) for i in whitePixLocations]
    arrY = [(lambda i: i[0]) (i) for i in whitePixLocations]
    if len(arrX) < 1 or len(arrY) < 1:
        return None
    return (int(np.median(arrX)), int(np.median(arrY)))

def make_video(image_list,name='silhouettes'):
  prefix = 'silhouette_'
  os.system('mkdir '+name)
  os.system('rm '+ data_dir + '/' +name+'/'+prefix+'*.png')
  last_existing = None
  i = 0
  for x in image_list:
    if x==None:
      x = last_existing
    else:
      last_existing = x
    if x==None:
      continue
    i = i + 1
    (dt,png, coords, coords3D, activity) = x
    image = png2image(png)
    if coords is not None:
        if not os.path.exists(name + '/' + activity):
            os.makedirs(name + '/' + activity)
        image_file = name + '/' + activity+ '/' + prefix + dt + 'marked.png'
        # Skaleerime pildi
        picSizePx = 32
        picSizeUnits = 2500
        image = extract_image_with_corresponding_values(image, coords, coords3D, picSizePx, picSizeUnits)
        if image is not None:
            cv2.imwrite(image_file, image)
  os.system('rm '+name+'.mp4')
  os.system('ffmpeg -framerate 30 -r 30 -i '+ data_dir + '/' +name+'/'+prefix+'%05d.png -c:v libx264 -pix_fmt yuv420p '+name+'.mp4')


data_dir = None
vid_file = None

def load_silhouettes():
  res = []
  frame_results = []
  data = None
  with open(data_dir + '/' + vid_file, mode='r') as data_file:
      data = data_file.read()
      for line in data.split("\n"):
          try:
              data_line = json.loads(line)
          except:
              pass
          if (data_line is not None):
              if len(data_line['e']) != 1:
                  coords = data_line['e'][2]['v']
                  coords3D = data_line['e'][4]['v']
                  activity = data_line['e'][6]['v']
                  dt = list(data_line['bt'].values())[0]
                  frame_results.append((dt, coords, coords3D, activity))
      for line in data.split("\n"):
          try:
              data_line = json.loads(line)
          except:
              pass
          if (data_line is not None):
              if len(data_line['e']) == 1:
                  silh = data_line['e'][0]['v']
                  dt = list(data_line['bt'].values())[0]
                  png = base64.b64decode(silh)
                  isFrameInResults = False
                  for frame in frame_results:
                      if frame[0] in dt:
                          coords = frame[1]
                          coords3D = frame[2]
                          activity = frame[3]
                          res.append((dt, png, coords, coords3D, activity))
  return(res)

make_video(load_silhouettes(),'')
