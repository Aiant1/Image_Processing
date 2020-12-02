# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 02:40:30 2020

@author: Antika

"""

import numpy as np
import cv2
import sys
import math
import time

from tkinter import *
def sigma(text):
    
    def close_window():
        global entry
        entry = E.get()
        entry=entry.split(" ")
#        print (entry)
        root.destroy()            
    root = Tk()
    E =Entry(root)
    E.pack(anchor = CENTER)
    B = Button(root, text =text,bg= 'dark green',height = 1, width =50,command = close_window)
    B.pack(anchor = S)
    root.mainloop()
    return entry


def gaussian_range(x,y,sigma_r):
     g_range=np.exp(-((x**2 + y**2)/(2.0*sigma_r**2)))
     return g_range
def gaussian_spatial(x,y,i,j,sigma_s):

     g_spatial=np.exp(-(((x-i)**2 + (y-j)**2)/(2.0*sigma_s**2)))
     return g_spatial

#image=image[0:8,0:7]

def bilateral (input_img,kernal_size,sigma_s,sigma_r):
    rows=input_img.shape[0]
    col=input_img.shape[1]
    half_kernal_size=kernal_size//2
    out_image = np.zeros(input_img.shape)
    
    
    for i in range(half_kernal_size,rows-half_kernal_size):
        for j in range(half_kernal_size,col-half_kernal_size):
            gs=0
            gr=0
            weight=0
            bilateral_filter=0
            central_point=[i,j]
            central_point_image_intensity=input_img[i,j]
    #        print ("cen: ",central_point)
            for x in range(i-half_kernal_size,i+half_kernal_size+1):
                for y in range(j-half_kernal_size,j+half_kernal_size+1):
    #                print ([x,y])
                    gs=gs+gaussian_spatial(i,j,x,y,sigma_s)
                    gr=gr+gaussian_range(input_img[x,y],central_point_image_intensity,sigma_r)
                    weight=weight+gs*gr
                    bilateral_filter=bilateral_filter+gs*gr*input_img[x,y]
                    bilateral_filter_normalized=bilateral_filter/weight
                    out_image[i,j]=bilateral_filter_normalized
    return out_image
             
def guided_bilateral (guided_img,depth_image,kernal_size,sigma_s,sigma_r):
    rows=guided_img.shape[0]
    col=guided_img.shape[1]
    half_kernal_size=kernal_size//2
    scale_row=rows//depth_image.shape[0]
    scale_col=col//depth_image.shape[1]
    out_image = np.zeros(guided_img.shape)
    
    
    for i in range(half_kernal_size,rows-half_kernal_size):
#        print (i/scale_row)
        for j in range(half_kernal_size,col-half_kernal_size):
            gs=0
            gs1=0
            gr=0
            weight=0
            bilateral_filter=0
            central_point=[i,j]
            central_point_image_intensity=guided_img[i,j]
#            central_point_image_intensity_low=depth_image[int(i/scale_row),int(i/scale_col)]
    #        print ("cen: ",central_point)
            for x in range(i-half_kernal_size,i+half_kernal_size+1):
                for y in range(j-half_kernal_size,j+half_kernal_size+1):                    
                    gs=gs+gaussian_spatial(int(i/scale_row),int(j/scale_col),int(x/scale_row),int(y/scale_col),sigma_s)
                    gr=gr+gaussian_range(guided_img[x,y],central_point_image_intensity,sigma_r)
                    weight=weight+gs*gr
                    bilateral_filter=bilateral_filter+gs*gr*depth_image[int(x/scale_row),int(y/scale_col)]
                    bilateral_filter_normalized=bilateral_filter/weight
            out_image[i,j]=bilateral_filter_normalized
##                    break
    return out_image            

     
if __name__ == "__main__":
    
    input_choice=input("please enter 1 for bilateral_filter or 2 for guided bitaleral  ")
    if input_choice=="2":
        start = time.time()

        image_guided=cv2.imread("image//image1.png")
        print (image_guided.shape)
        image_guided=image_guided[:,0:1389]
            
        c1 = image_guided[:,:,0]
        c2 = image_guided[:,:,1]
        c3 = image_guided[:,:,2]
        image_guided=c1+c2+c3
        depth_map=cv2.imread("image//imagedisp1.png")
        print (depth_map.shape)
        depth_image=cv2.cvtColor(depth_map,cv2.COLOR_BGR2GRAY)
        sigma_s=sigma("enter sigma s value")
        sigma_r=sigma("enter sigma r value")
        k=3
        for i in sigma_s:
            for j in sigma_r:
                print (type(i))
                print (type(j))
                opt=guided_bilateral(image_guided,depth_image,k,int(i),int(j))
                out="image//"+"image2"+str(k) + i +j +".png"
                cv2.imwrite(out,opt)
                cv2.imshow("out",opt)
        end=time.time()
        print(f"Runtime of the program is {end - start}")

    if input_choice=="1":
        
        start=time.time()
        image = cv2.imread("image//geometry.jpeg")
        image_bilateral = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sig_s=sigma("enter values for sigma_s")
        sig_r=sigma("enter values for sigma_r")
        kernal_size=9
        print ("bilateral execution is in process")
        if len(sig_s)==len(sig_r):
            for i in sig_r:
                for j in sig_s:
                    output=bilateral(image_bilateral,kernal_size,int(j),int(i))
                    string="image//"+"geometry"+"_"+str(kernal_size)+"_"+str(i)+"_"+str(j)+".jpg"
                    cv2.imwrite(string,output)
                    print ("saved")
        else:
              print ("sigma s and sigma _r values should be equal")
        
        print ("bilateral process is done")
        end=time.time()
        print(f"Runtime of the program is {end - start}")

        

#

        
        
        
  


                
                
                
                
    













