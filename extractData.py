import os
import cv2
import numpy as np

# Kasuta python 2.7.10
def png2image(png):
      return(cv2.imdecode(np.fromstring(png,np.uint8),cv2.IMREAD_COLOR))

def extract_img(image, coords, coords3D,
                const2Dsize, const3Dvalue):
    # Conversion on how many units in 3d is 1 pixel in 2d
    conversion = const3Dvalue / const2Dsize
    # Calculate new sizes
    newWidth = convertSize(abs(coords3D[3] - coords3D[0]), conversion)
    newHeight = convertSize(abs(coords3D[4] - coords3D[1]), conversion)
    if newWidth != 0 and newHeight != 0:
        # Crop the image so it could be easier to resize
        croppedImage = image[coords[1]:coords[3], coords[0]: coords[2]]
        # Resize the image
        modImg = cv2.resize(croppedImage, (newWidth, newHeight),
                            interpolation = cv2.INTER_AREA)
        # Create a new image and position croppedImage in the middle
        extractedImg = np.zeros((const2Dsize, const2Dsize, 3), np.uint8)
        height, width, channel = modImg.shape
        if height is not None and width is not None:
            paddingX = int((const2Dsize - width)/2)
            paddingY = int((const2Dsize - height)/2)
            if paddingX > 0 and paddingY > 0:
                for y in range(0, height):
                    for x in range(0, width):
                        extractedImg[paddingY+y, paddingX+x] = modImg[y, x]
                return extractedImg
    return None

def convertSize(diff3D, conversion):
    return int(round((diff3D/conversion), 0))

#Filtreerib ja tagastab mediaan väärtuse
#crds - koordinaadid

def filterOutOuterValuesAndReturnCenter(coordList, crds):
    arrX = []
    arrY = []
    for crd in coordList:
        if (crd[1] > min(crds[0], crds[2])) and (crd[1] < max(crds[0], crds[2])) \
                and (crd[0] > min(crds[1], crds[3])) and (crd[0] < max(crds[1], crds[3])):
            arrX.append(crd[1])
            arrY.append(crd[0])
    if len(arrX) < 1 or len(arrY) < 1:
        return None
    return (int(np.median(np.sort(arrX))), int(np.median(np.sort(arrY))))

def make_pictures(image_list, name='silhouettes'):
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
        # Saab valgete pikslite asukohad
        npImg = np.asarray(image)
        coordList = np.argwhere(npImg == [255, 255, 255])
        #Leiame valgete piklsite mediaani
        centerLoc = filterOutOuterValuesAndReturnCenter(coordList, coords)
        if not os.path.exists(name + '/' + activity):
            os.makedirs(name + '/' + activity)
        image_file = name + '/' + activity+ '/' + prefix + dt + 'marked.png'
        #Kui pole keskpunkti olemas, siis ära tee midagi
        if centerLoc is not None:
            # Skaleerime pildi ( endale soovitud pikslite
            picSizePx = 32
            picSizeUnits = 2500
            image = extract_img(image, coords, coords3D,
                                picSizePx, picSizeUnits)
            # Kui kõik korras, siis loo pilt
            if image is not None:
                cv2.imwrite(image_file, image)


data_dir = None # <--- andmete asukoht
# probleem et reanumbrid olid ka sees
# parandamiseks cut -f 2- -d: VID_bsondump_04_30_and_05_01.txt > VID_bsondump_04_30_and_05_01_corrected.txt
vid_file = None # <-- faili nimetus

def load_silhouettes():
  res = []
  frame_results = []
  data = None
  with open(data_dir + '/' + vid_file, mode='r') as data_file:
      data = data_file.read()
      # Esimene iteratsioon leiab read, kus on piltide kohta käivad andmed
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
      # Teine iteratsioon leiab pildi kodeeritud kuju ja otsib ülesse pildi ajatempli järgi ning lisab lõplikku
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
                  for frame in frame_results:
                      if frame[0] in dt:
                          coords = frame[1]
                          coords3D = frame[2]
                          activity = frame[3]
                          res.append((dt, png, coords, coords3D, activity))
  return(res)
# Teine parameeter on kausta nimetus
make_pictures(load_silhouettes(), None)