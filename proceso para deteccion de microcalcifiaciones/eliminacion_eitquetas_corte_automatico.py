import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import glob

def most_common(mamo,ROW,COLUMN):
    values , counts = np.unique(mamo[:,:,0], return_counts=True)
    values = np.delete(values,0)
    counts = np.delete(counts,0)
    index = np.where(counts == np.amax(counts))
    for x in range(ROW):
        for y in range(COLUMN):
            if(values[index]==mamo[x,y,0]):# assign the white color to the pixels with the colorvalue more common
                mamo[x,y]=[255,255,255]
            else:
                mamo[x,y]=[0,0,0]# in otherwise the color value is black
    return mamo

def result_cut(Mb,Mo,ROW,COLUMN):
    for index in range (0,COLUMN):
        if(np.any(Mb[:,index] == 255)):
            firstColumn = index
            break

    for index in range(COLUMN-1,-1,-1):
        if(np.any(Mb[:,index] == 255)):
            lastColumn = index
            break

    COLUMN=lastColumn-firstColumn #assign the accurate column
    Mc=np.zeros((ROW,COLUMN,3), dtype = np.uint8)
    for row in range(ROW):
        index=-1
        for col in range(firstColumn,lastColumn):
            index+=1
            if(np.any(Mb[row,col]==0)):
                Mo[row,col]=[0,0,0]
            Mc[row,index]=Mo[row,col] # assing values

    return Mc

def smoothing_edges(mamografy):
    
    retval, threshold = cv2.threshold(mamografy, 20, 255, cv2.THRESH_BINARY) # to convert a binary image, if the values are (12-255) then the pixel value will be white, in otherwise will be black
    ROW, COLUMN = threshold.shape[:2] #getting dimesionals values
    
    threshold[0:20,0:COLUMN] = [0,0,0] # converting in black that RIO
    threshold[(ROW-20):ROW,0:COLUMN] = [0,0,0] # converting in black that RIO
    
    border = cv2.GaussianBlur(threshold, (5,5), 10)
    border2 = cv2.GaussianBlur(border, (5,5), 10) # taking borders to borders
    
    retval, threshold = cv2.threshold(border2, 12, 255, cv2.THRESH_BINARY) # binary beacuse the borders sometimes has intesity diferrent to 255
    kernel = np.ones((8,8),np.uint8)
    erosion = cv2.erode(threshold,kernel,iterations = 1) # eroting to delete expanded noise 
    
    return erosion, ROW, COLUMN

if __name__=='__main__':
    
    PATH = glob.glob("../imgs/mamografia MIAS/*.pgm") # getting the route and the name of each file (.pgm) this is save in a array
    PATH = np.asarray(PATH)
    contador = 300
    
    for img in PATH[(contador):]: # to do this procces automatically in all imgs of the data base MIAS
        
        mammography = cv2.imread(img)
        
        smoothing, ROW, COLUMN = smoothing_edges(mammography) # to smooth the border of the binary image
        
        labeled_coins,_ = ndi.label(smoothing) # to labeled each white region
        
        most_commonn = most_common(labeled_coins,ROW,COLUMN) # to let the color most_common in labaled_mamo
        
        result_cutt = result_cut(most_commonn,mammography,ROW,COLUMN) # to cut the image to only have mammography
        
        plt.imshow(result_cutt)
        plt.show()
        
        contador += 1
        print(contador)
        contador = str(contador)
        cv2.imwrite("../imgs/mamografia MIAS recortadas/mamo" + contador + ".jpg",result_cutt)
        contador = int(contador)
        