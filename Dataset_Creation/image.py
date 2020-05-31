import cv2
import csv
print( cv2.__version__ )
j=0
def csvmaker(s,x):
    fd = open('image.csv', 'a')
    myCsvRow=s+','+x+'\n'
    print(myCsvRow)
    fd.write(myCsvRow)
    
    fd.close()
def imageprocess(i):
    img = cv2.imread('test.jpg') #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    for x in range(0, len(hsv)):
        for y in range(0, len(hsv[0])):
            hsv[x, y][2] = i+150

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    s='test'+str(i)+'.jpg'
    cv2.imwrite(s, img)
def imageprocess2(i):
    img = cv2.imread('test.jpg') #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
      
    v += i
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    j=50+i
    print(hsv[20,20][2])
    s1='test'+str(j)+'.jpg'
    cv2.imwrite(s1, img)
    img = cv2.imread(s1) #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)

    x=str(hsv[6,6][2])+','+str(hsv[50,50][2])+','+str(hsv[100,100][2])+','+str(hsv[150,150][2])+','+str(hsv[200,200][2])+','+str(hsv[250,250][2])
    csvmaker(s1,x)    

def imageprocess3(i):
    img = cv2.imread('test.jpg') #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    v -= i
    print(hsv[20,20][2])
    j=50-i+1
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    s1='test'+str(j)+'.jpg'
    cv2.imwrite(s1, img)
    img = cv2.imread(s1) #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    x=str(hsv[6,6][2])+','+str(hsv[50,50][2])+','+str(hsv[100,100][2])+','+str(hsv[150,150][2])+','+str(hsv[200,200][2])+','+str(hsv[250,250][2])
    csvmaker(s1,x)

myCsvRow='Brightness_In_pixel(6,6)'+','+'Brightness_In_pixel(50,50)'+','+'Brightness_In_pixel(100,100)'+','+'Brightness_In_pixel(150,150)'+','+'Brightness_In_pixel(200,200)'+','+'Brightness_In_pixel(250,250)'+'\n'
csvmaker('Image_Name',myCsvRow)
for c in range (50,1,-1):
    imageprocess3(c)
for c in range (0,51):
    imageprocess2(c)

