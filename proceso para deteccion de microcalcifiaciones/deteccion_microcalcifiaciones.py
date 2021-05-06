import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy import ndimage as ndi
from collections import OrderedDict
from skimage import feature

def most_common(mamo,delete):
    values , counts = np.unique(mamo[:,:], return_counts=True) # getting the counts accordint to the pixel values
    values = np.delete(values,0)
    counts = np.delete(counts,0)
    index = np.where(counts == np.amax(counts)) # here we know what is the pixel mosth common
    if(delete == 0): # to delete the small regions
        for x in range(mamo.shape[0]):
            for y in range(mamo.shape[1]):
                if(values[index]==mamo[x,y]):# assign the white color to the pixels with the colorvalue most common
                    mamo[x,y] = 255
                else:
                    mamo[x,y] = 0# in otherwise the color value is black
    else: # to delete the big region
        for x in range(mamo.shape[0]):
            for y in range(mamo.shape[1]):
                if(values[index]==mamo[x,y]):# assign the white color to the pixels with the colorvalue more common
                    mamo[x,y] = 0
                
    return mamo

def result_cut(Mb,Mo,ROW,COLUMN):
    for index in range (0,COLUMN): # to get the first column that contain the mammography
        if(np.any(Mb[:,index] == 255)):
            firstColumn = index
            break

    for index in range(COLUMN-1,-1,-1): # to get the last column that contain the mammography
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
                Mo[row,col]=0
            Mc[row,index]=Mo[row,col] # assing values

    return Mc

def smoothing_edges(mamografy):
    
    retval, threshold = cv2.threshold(mamografy, 20, 255, cv2.THRESH_BINARY) # to convert a binary image, if the values are (12-255) then the pixel value will be white, in otherwise will be black
    ROW, COLUMN = threshold.shape[:2] #getting dimesionals values
    
    threshold[0:20,0:COLUMN] = 0 # converting in black that RIO
    threshold[(ROW-20):ROW,0:COLUMN] = 0 # converting in black that RIO
    
    border = cv2.GaussianBlur(threshold, (5,5), 10)
    border2 = cv2.GaussianBlur(border, (5,5), 10) # taking borders to borders
    
    retval, threshold = cv2.threshold(border2, 12, 255, cv2.THRESH_BINARY) # binary beacuse the borders sometimes has intesity diferrent to 255
    kernel = np.ones((8,8),np.uint8)
    erosion = cv2.erode(threshold,kernel,iterations = 1) # eroting to delete expanded noise 
    
    return erosion, ROW, COLUMN

def corection_gamma(mamo): # here we apply the power law in the mammography to delete great part of  fatty tissue
    mamo = mamo.astype(np.float32) 
    mamo = mamo/255
    mamo = np.power(mamo,5) # watch the graph
    mamo = mamo*255
    mamo = mamo.astype(np.uint8) 
    return mamo

def intensityValues_acommodation(corection_gammaa):
    
    intensityValues , nPixels = np.unique(corection_gammaa, return_counts=True) 
    nPixels[0:50] = 0 # deleting this data because therea are no important to the prediction model because is data of the background
    
    index = 0
    values = np.zeros(220)
    for intensity in range(220): # to acommodate the how much Npixels conrrespons to each pixel (0-220)
        
        if((intensity == intensityValues[index])):
            values[intensity] = nPixels[index]
            if(intensity < intensityValues[-1]):
                index +=1
        else:
            values[intensity] = 0
        
    return values

def density_clasification(PI, totalPixels, mamoModel):
    
    mamoFeatures = OrderedDict([ # acommodating the data neccesary to the prediction
        ('P1', np.mean(PI[50:99])),
        ('Std1', np.std(PI[50:99])),
        ('P2', np.mean(PI[99:150])),
        ('Std2', np.std(PI[99:150])),
        ('N2', (np.sum(PI[99:150])/totalPixels))
    ])
    
    mamoFeatures = pd.Series(mamoFeatures).values.reshape(1,-1) # to convert in a 2-D array
    
    denseTissue = float(mamoModel.predict(mamoFeatures)) # getting the prediction
    
    # print(denseTissue)
    
    if denseTissue <= 0.46: # among nearer to 1 more dense 
        Tissue = 'fatty tissue'
    if denseTissue>0.46:
        Tissue = 'dense tissue'
        
    return Tissue


    
def microcalcification_fattyTissue(mamo):

    mamoOriginal = cv2.cvtColor(np.uint8(mamo), cv2.COLOR_GRAY2BGR)
    
    ''' to thesholding only the microcalcifiactions '''
    intensityValues , nPixels = np.unique(mamo, return_counts=True)
    mostCommon = int(np.sum(intensityValues[50:]*nPixels[50:])/np.sum(nPixels[50:])) + 1 # most common pixel
    retval, threshold = cv2.threshold(corection_gammaa, mostCommon, 255, cv2.THRESH_BINARY)
    
    ''' to wrap the pectoral muscle (white) to delete it '''
    
    if(np.any(threshold[:,-1] == 255)):
        for column in range(threshold.shape[1]-1,-1,-1):
            if(np.all(threshold[:,column] == 0)):
                threshold[21:400,(column-10):-1] = 255
                break
            
    if(np.any(threshold[:,0] == 255)):
        for column in range(0,threshold.shape[1]):
            if(np.all(threshold[:,column] == 0)):
                threshold[21:400,0:(column+10)] = 255
                break
    
    ''' to only let the white regions of the possible microcalcificactions '''
    
    kernel = np.ones((4,4),np.uint8)
    dilate = cv2.dilate(threshold,kernel,iterations = 1)
    labeled_mamo,_ = ndi.label(dilate)
    threshold = most_common(labeled_mamo,1)
    threshold = threshold > 0
    
    ''' to assign red color to he area where is the microcalcifaction in the mammography '''
    
    for row in range(threshold.shape[0]):
        for column in range(threshold.shape[1]):
            if(np.any(threshold[row,column] == 1)):
                mamoOriginal[row,column] = [255,0,0]
    
    return mamoOriginal

if __name__ == '__main__':
    
    mamoModel = joblib.load('machine learning/mammography_model.pkl')
    PATH = glob.glob("imgs/mamografia MIAS/*pgm") # getting the N files jpg in the route 
    # contador = 0
    for img in PATH[5:11]:
        
        ''' mammography segmentation '''
        
        mammography = cv2.imread(img)
            
        smoothing, ROW, COLUMN = smoothing_edges(mammography[:,:,0]) # to smooth the border of the binary image
        
        labeled_mamo,_ = ndi.label(smoothing) # to labeled each white region
        
        most_commonn = most_common(labeled_mamo,0) # to let the color most_common in labaled_mamo
        
        result_cutt = result_cut(most_commonn,mammography,ROW,COLUMN) # to cut the image to only have mammography
        
        ''' data to the model prediction '''
    
        corection_gammaa = corection_gamma(result_cutt[:,:,0]) # to delete the  fatty tissue
        
        intensityValues = intensityValues_acommodation(corection_gammaa) # to acommodate the intesityValues (0-220) to use it in the prediction model 
        
        tissue = density_clasification(intensityValues, np.sum(intensityValues), mamoModel) # to clasificate the mammography 
        
        ''' to search microcalcifications in mammography '''
        
        if(tissue == 'fatty tissue'):
            microcalcifications = microcalcification_fattyTissue(corection_gammaa)
        if(tissue == 'dense tissue'):
            microcalcifications = corection_gammaa
        
        ''' showing the info '''
        
        plt.subplot(1,2,1)
        plt.imshow(corection_gammaa,'gray')
        plt.title(tissue)
        
        plt.subplot(1,2,2)
        plt.imshow(microcalcifications)
        plt.title('Microcalcificaciones')
        plt.show()