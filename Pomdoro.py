# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:17:36 2021

@author: Fraser

Pomodoro executable

TO DO:
Goal: Exe file, basic gui with user input for message and drop down for timer options 
Work period start/end 
work period time/rest time 
Pause option

Bonus adds: Royalty free music, warning of impending work/rest phase
Nicer GUI
Inspirational text messages etc
Make it look/feel professional
Upload to github with accompanying screen shots for portfolio

"""

import winsound
import pyttsx3
import time
import keyboard #Using module keyboard
import sys
#vars

strtWork = "The work period starts now, for "
endWork = "The work period is pausing, take a break for: "
almostOver = "Period is finishing in: "
m = 60
workTime = int(30*m)  # in seconds
restTime = int(10*m)
warningTime = 0.9
global repeats
repeats =6


def onRun(sw, ew, almOvr,  wt, rt, wrnt, rpts):
    #BASIC GUI, SET TIMER REPEAT LENGTH, SET BREAK LENGTH
    #Input message for voice engine to speak start/end work period
    workPeriod(sw, ew, almOvr,  wt, rt, wrnt,rpts)
  
    
    
def soundEngine(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait() 


def warningMessage(sw, ew, almOvr,  wt, rt, wrnt):  # move warning message into here, if less than 1 minutes, use seconds. 
    pass

def workPeriod(sw, ew, almOvr,  wt, rt, wrnt, rpts):
    print("Work Period")
    # CHECK FOR TYPE ERRORS
    warned = False
    soundEngine(sw+str(wt/60)+"minutes, repeat number: " + str(rpts))
    strtTime = time.time()
    while True:  #making a loop
        currTime = time.time()
       
        timeEl = int(currTime -strtTime)
        if(timeEl%(wt*0.1)==0):
           print("Working for: ", timeEl/60)
           time.sleep(1)
        if currTime -strtTime >= wt:
            soundEngine(ew+str(wt/60)+"minutes")
            if rpts >= repeats:
                soundEngine("Work finished!")
                sys.exit()
            restPeriod(sw, ew, almOvr,  wt, rt, wrnt, rpts)
            break
        if (currTime -strtTime > wt*wrnt and currTime -strtTime <= wt and warned == False):
            warned = True
            timeLeft = str(round((wt-(wrnt*wt))/60,2))
            soundEngine(almOvr+timeLeft+"minutes")
        try:  #used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('esc'): #if key 'a' is pressed 
                print('You Pressed A Key!')
                sys.exit()#finishing the loop
            else:
                pass
        except:
            break


def restPeriod(sw, ew, almOvr,  wt, rt, wrnt,rpts):
     print("Rest Period")
     strtTime = time.time()
     warned = False
     while True:  #making a loop
        currTime = time.time()
        
        
        timeEl = int(currTime -strtTime)
        if(timeEl%(0.1*rt)==0):       
            print("resting for: ", timeEl/60)
            time.sleep(1)
            
        if currTime -strtTime >= rt:
            rpts+=1
            workPeriod(sw, ew, almOvr,  wt, rt, wrnt, rpts)
            break
        if (currTime -strtTime > rt*wrnt and currTime -strtTime <= rt and warned == False):
            warned = True
            timeLeft = str(round((wrnt*rt)/60,2))
            soundEngine(almOvr+timeLeft+"minutes")
        try:  #used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('esc'): #if key 'a' is pressed 
                print('You Pressed A Key!')
                sys.exit()#finishing the loop
            else:
                pass
        except:
            break
    
    
onRun(sw = strtWork, ew = endWork,  almOvr=almostOver, wt = workTime, rt = restTime, wrnt = warningTime,rpts = 0)
