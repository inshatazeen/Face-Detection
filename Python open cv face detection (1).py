#!/usr/bin/env python
# coding: utf-8

# ## Open Cv face detection
# ## Import open cv and harrcascade

# In[4]:


pip install opencv-python


# In[5]:


import cv2


# In[6]:


faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# ### Read image and convert it into grayscale

# In[9]:


image=cv2.imread("istockphoto-1092715214-170667a.jpg") 


# In[10]:


from matplotlib import pyplot as plt
plt.imshow(image)


# In[12]:


RGB_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img)


# In[17]:


GR_img=cv2.cvtColor(RGB_img,cv2.COLOR_BGR2GRAY)
plt.imshow(GR_img)


# ## Detect Cordinates of faces

# In[32]:


faces= faceCascade.detectMultiScale(
    GR_img,
     scaleFactor=1.1,
     minNeighbors=5,
     minSize=(30,30)
)
print(faces)


# In[30]:


image=cv2.imread("istockphoto-1092715214-170667a.jpg")
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),20)
    RGB_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)


# In[31]:


ims=[]
for (x,y,w,h) in faces
crop=RGB_img[y:y+h,x:x+w]
ims.append(crop)


# In[ ]:





# 
