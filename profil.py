#!/usr/bin/env python 


#Skript ist momentan nur rudimentaer und unausgemistet, sorry!



from PIL import Image
from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
import numpy as np

#todo: calculate gauss-filter with s defined by the command line
#v=[1,3,6,9,10,9,6,3,1] # Gauss Filter with s=2
v=[1,2,4,7,10,13,16,17,16,13,10,7,4,2,1] # Gauss Filter with s=3


#todo: define image file in command line

#todo: check if laser beam is horizontal or vertical in the image

#todo: delete things in the image that are not the laser beam

#todo: detect laser beam in a non-dark image background (e.g. open two images of the same background, one with laser beam, one without. correlate them and subtract them so that only the laser beam is left)

#img=Image.open("/home/martin/Bilder/Laser/IMG_20150422_203922b.jpg")
img=Image.open("/home/martin/Bilder/Laser/DSC_8243b.jpg")
#img=Image.open("/home/martin/Bilder/Laser/DSC_8207_mod.jpg")
#img=Image.open("/home/martin/Bilder/Laser/DSC_8207.JPG")
r,g,b=img.split()
pix=r.load()
profile=[]
profilec=[]
profilef=[]
pvalue=[]
A=np.asarray(r)
for x in xrange(r.size[0]):
	profile.append(0)
	profilec.append(0)
	pvalue.append(0)
	profilef.append(0)
	#print(profile)
	col=A[:,x]
	colc=np.convolve(A[:,x],v,mode='same')
        pvalue[x]=max(colc)
        profile[x]=np.argmax(col)
        profilec[x]=np.argmax(colc)
        
        #parabolic sub-pixel maximum finder (http://www-ist.massey.ac.nz/dbailey/sprg/pdfs/2003_IVCNZ_414.pdf)
        if ( profilec[x] > 0 and profilec[x] < r.size[1]):
	  profilef[x] = (colc[profilec[x]+1] - colc[profilec[x]-1]) / (4.0*colc[profilec[x]] - 2.0*(colc[profilec[x]+1] + colc[profilec[x]-1] ))
	profilef[x] = profilef[x] + profilec[x]
	
	#todo:
	#zero-filling and filtering sub-pixel maximum finder
	#oversampling_factor = 8 	# 1 value, n-1 zeros
	#searchrange = 10		# search n pixels before and n after the maximum value
	
	
	
#        for y in xrange(r.size[1]):
#	  if pix[x,y] > pvalue[x]:
#	    profile[x] = y
#	    pvalue[x] = pix[x,y]

#todo: 
#geometrische Anpassung, wenn Brennweite bekannt

xaxis = np.arange(r.size[0])
profilefnp=np.asarray(profilef)
preselector = pvalue > np.mean(pvalue)*0.75 

#todo: define the pixels that should not be used in the polyfit (e.g. the reference object)

#linear fit to the data to get a straight line of the residuals as result
p = np.polyfit( xaxis[preselector], profilefnp[preselector], 1,)
flattened=profilef-p[1]-p[0]*xaxis

#quadratic fit to the data to get a straight line of the residuals as result

#p = np.polyfit( xaxis[preselector], profilefnp[preselector], 2,)
#flattened = profilef - p[2] - p[1]*xaxis - p[0] * xaxis**2


#todo: variable high pass filtering to remove lens errors or waviness


#points with weak lighting are marked

weak1=flattened*1.0
weak2=flattened*1.0

weak1[pvalue > (np.mean(pvalue)*0.5)]=np.nan
weak2[pvalue > (np.mean(pvalue)*0.75)]=np.nan
#flattened[pvalue < (np.mean(pvalue)*0.7)]=np.nan

#np.place(weak1,pvalue < (np.mean(pvalue)/3),None)
#np.place(weak2,pvalue < (np.mean(pvalue)/2),None)

#cleaned=profilef
#cleaned[pvalue < (np.mean(pvalue)/2)]=np.nan

#warning if too many overdriven pixels
overdriven=np.count_nonzero(A>254)
print overdriven
allpixels=r.size[0]*r.size[1]
overdrivenpercent = 100.0* overdriven / allpixels

print ('%i pixels are at limit! (%f percent)' %(overdriven,overdrivenpercent))

	
#figure1: raw profile data of various maximum finding algorithms
plt.figure(1) 
#ax = SubplotZero(fig, 111)
#fig.add_subplot(ax)
plt.plot(profile,'r')
plt.plot(profilec)
plt.plot(profilef,'g')
plt.title('raw profile data of various maximum finding algorithms')
plt.grid(b=True, which=u'major')
#plt.plot(clean,'g')

#figure2: illumination data
plt.figure(2) 
#plt.plot(A[:,20],'r')
#plt.plot(np.convolve(A[:,20],v,mode='same')/sum(v))#,mode='same')
plt.plot(pvalue/np.sum(v))
plt.title('illumination data')
plt.grid(b=True, which=u'major')

#figure3: filtered profile in pixels
plt.figure(3)
plt.plot(xaxis,flattened,'g')
plt.plot(xaxis,weak2,'y')
plt.plot(xaxis,weak1,'r')
plt.grid(b=True, which=u'major')
plt.xlabel('length / px')
plt.ylabel('height / px')
plt.title('filtered profile in pixels')
#plt.plot(xaxis, pvalue < (np.mean(pvalue)/3),'r')
#plt.plot(xaxis, pvalue < (np.mean(pvalue)/2),'y')

#todo in far future: recognize the reference object and its dimensions in pixels in the image
height_px = 20
width_px = 50

#todo: define the values of the reference object in the command line
height_mm = 23.0
width_mm = 37.0

x_scale = width_px / width_mm # pixel pro mm breite
y_scale = height_px / height_mm # pixel pro mm hoehe


#figure3: filtered profile in mm
plt.figure(4)
plt.plot(xaxis / x_scale, flattened / y_scale,'g')
plt.plot(xaxis / x_scale, weak2 / y_scale,'y')
plt.plot(xaxis / x_scale, weak1 / y_scale,'r')
plt.grid(b=True, which=u'major')
plt.xlabel('length / mm')
plt.ylabel('height / mm')
plt.title('filtered profile in mm')
plt.show() 

#todo: calculate roughness parameters like Rz, Ra, Rq...


#f = open('/home/martin/Bilder/Laser/DSC_8207_mod.txt','w')
#for k in profile:
#  f.write("%s,%s\n" %(profile[k], pvalue[k]) )
#f.close
