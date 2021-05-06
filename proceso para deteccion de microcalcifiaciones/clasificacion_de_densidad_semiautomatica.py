import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def corection_gamma(mamo): # here we apply the power law in the mammography to delete great part of  fatty tissue
    mamo = mamo.astype(np.float32) 
    mamo = mamo/255
    mamo = np.power(mamo,5) # watch the graph
    mamo = mamo*255
    mamo = mamo.astype(np.uint8) 
    return mamo

def intensityValues_acommodation(intensityValues,nPixels):
    index = 0
    values = np.zeros(220)
    for intensity in range(220):
        
        if((intensity == intensityValues[index])):
            values[intensity] = nPixels[index]
            if(intensity < intensityValues[-1]):
                index +=1
        else:
            values[intensity] = 0
        
    return values

if __name__=='__main__':
    
    PATH = glob.glob("../imgs/mamografia MIAS recortadas/*jpg") # getting the N files jpg in the route 
    contador = 0
    
    mamoData = pd.DataFrame(columns = ['P1','Std1','N1','P2','Std2','N2','P3','Std3','N3','Tissue'])
    
    
    for img in PATH:
        mammography = cv2.imread(img)
        mammography = mammography[:,:,0] # to only use the first canal
        
        plt.subplot(2,2,1)
        plt.imshow(mammography,'gray')
        plt.axis("off")
        
        corection_gammaa = corection_gamma(mammography) # to apply the power law in the mammography
        plt.subplot(2,2,2)
        plt.imshow(corection_gammaa,'gray')
        plt.axis("off")

        intensityValues , nPixels = np.unique(mammography, return_counts=True) # to mammography
        nPixels[0] = 0
        
        plt.subplot(2,2,3)
        plt.plot(intensityValues,nPixels) # plotting intensityValues vs nPixels
        plt.title('Sinus histogram')
        plt.xlabel('Intesity')
        plt.ylabel('Number of pixels')
        
        intensityValues , nPixels = np.unique(corection_gammaa, return_counts=True) # to correction_gamma
        nPixels[0:50] = 0
        
        # print(intesityValues)
        plt.subplot(2,2,4)
        plt.plot(intensityValues,nPixels)
        plt.title('Power law histogram')
        
        # data necessary to compute the data to the csv file (developing of the model)
        
        counts = intensityValues_acommodation(intensityValues,nPixels) # (PI) mean -->(pixelsIntensity) / i do this because the array intensityValues, has not the values ordenate and there are space between them ej: (148,151,152,...)
        totalPixels = np.sum(nPixels)
        
        plt.show()
        cv2.destroyAllWindows()
        
        ''' 
            here we are going to save the info of the power law hist in a csv file to design a machine 
            learning model to classify the breasts in fatty tissue or dense tissue, we are take in 
            consideration the following data:
            
            1-) mean / mean general to each inteval
            2-) standard desviation / std general to each interval
            3-) frecuency / total pixel (img) to each interval
    
            the intervals are (50-100, 100-150, 150-190)  !intesity¡
        '''
        
        tissue = float(input(str(contador) + "-) El tejido es denso?: ")) # 1 to yes / 0 to not
        
         
        
        
        mamoData.loc[contador] = [np.mean(counts[50:99]),np.std(counts[50:99]),(np.sum(counts[50:99])/totalPixels),
                                  np.mean(counts[99:150]),np.std(counts[99:150]),(np.sum(counts[99:150])/totalPixels),
                                  np.mean(counts[150:190]),np.std(counts[150:190]),(np.sum(counts[150:190])/totalPixels),tissue]

        contador += 1
        # if(contador == 50):
        #     break
        
    mamoData.to_csv('../csv/mamoData4.csv', index = False)