import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy import ndimage as ndi
import glob

def neighbour_N25(mamo,x,y,ROW,COLUMN):
    colorvalue = np.zeros((3)) #we define the matriz colorvalue
    for coorx in range(x, x+16):#we go through horizontally. with reference to the pixel white (x=i-1,y=j-1)
        for coory in range(y, y+16): # searching if the Vecinos25 have a color value different to white
            if((np.any((mamo[coorx,coory]!=255) & (mamo[coorx,coory]!=0)))):#if there is/are a color value diferrent to white in the Vecino25
                colorvalue = mamo[coorx,coory] #then colorvalue equal to color value of that pixel
                return colorvalue # new = False

    for index in range(3):# if all Vecinos25 are white then we gonna assing a color value
        colorvalue[index] = randint(1,253) # assing in value to each channel, between 10-230 (random)

    return colorvalue #new = True

def color_areas(mamo,ROW,COLUMN):
    for x in range(8,ROW-16): # we go through the matrix horizontally
        for y in range(8,COLUMN-16):
            if(np.any(mamo[x,y] == 255)):# if the pixel (i,j) is white then
                colorvalue = neighbour_N25(mamo,(x-8),(y-8),ROW,COLUMN) # we send parameters
                for coorx in range((x-8),x+16):
                    for coory in range((y-8),y+16):
                        if(((np.any(mamo[coorx,coory] == 255)))):# here we assing the colorvalue to each white pixel
                            mamo[coorx,coory] = colorvalue 
                
                
    return mamo

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
    
    PATH = glob.glob("../imgs/mamografia MIAS/*.pgm") #getting the route and the name of each file (.pgm) this is save in a array
    PATH = np.asarray(PATH)
    contador = 151
    
    for img in PATH[(contador):200]:
        
        mammography = cv2.imread(img)
        
        smoothing, ROW, COLUMN = smoothing_edges(mammography)
        
        color_areass = color_areas(smoothing,ROW,COLUMN) # we send the parameters 
        
        most_commonn = most_common(color_areass,ROW,COLUMN)
        
        result_cutt = result_cut(most_commonn,mammography,ROW,COLUMN)
    
        
        plt.imshow(result_cutt)
        plt.show()
        
        contador += 1
        print(contador)
        contador = str(contador)
        cv2.imwrite("../imgs/mamografia MIAS recortadas/mamo" + contador + ".jpg",result_cutt)
        contador = int(contador)
        