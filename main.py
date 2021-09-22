import threading
import time 
import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
global pridictValue,Thread1IsFinish,Thread2IsFinish,classNames,state,cx2,KiemTraSuXuatHien,tmp
Thread1IsFinish = 1
Thread2IsFinish = 1
pridictValue = 'Nothing'
state = 'Forward'
recentSign = 'Stop'
delayTime =1
cx2 = 320
KiemTraSuXuatHien = ['Nothing','Nothing','Nothing','Nothing','Nothing']
KienTraVuaRe = 0
tmp = 0
# Difine and setup GPIO pin
# Define my motor pin
# Left motor: 31 -33
# Right motor: 35 - 37
IN1 = 35
IN2 = 37
IN3 = 33
IN4 = 31
# set pin numbers to the board's
GPIO.setmode(GPIO.BOARD)

# initialize In1 In2 In3 and In4
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

''' For GPIO '''
# Stop
def Stop():
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN2, GPIO.LOW)
	GPIO.output(IN3, GPIO.LOW)
	GPIO.output(IN4, GPIO.LOW)


'''-------------------------------------'''
# Forward
def Forward():
	GPIO.output(IN1, GPIO.HIGH)
	GPIO.output(IN2, GPIO.LOW)
	GPIO.output(IN3, GPIO.HIGH)
	GPIO.output(IN4, GPIO.LOW)

	
'''-------------------------------------'''
# Backward
def Backward():
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN2, GPIO.HIGH)
	GPIO.output(IN3, GPIO.LOW)
	GPIO.output(IN4, GPIO.HIGH)
'''-------------------------------------'''

#Left
def Left(liDo):
	if liDo >170 and liDo <470:
		GPIO.output(IN1, GPIO.LOW)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.HIGH)
		GPIO.output(IN4, GPIO.LOW)
		time.sleep(0.03)
		GPIO.output(IN1, GPIO.HIGH)
		time.sleep(0.01)
	else:
		GPIO.output(IN1, GPIO.LOW)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.HIGH)
		GPIO.output(IN4, GPIO.LOW)
		time.sleep(0.07)
		GPIO.output(IN1, GPIO.HIGH)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.HIGH)
		GPIO.output(IN4, GPIO.LOW)
		time.sleep(0.01)
	
'''-------------------------------------'''
#Right
def Right(liDo):
	if liDo >170 and liDo <470:
		GPIO.output(IN1, GPIO.HIGH)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.LOW)
		GPIO.output(IN4, GPIO.LOW)
		time.sleep(0.02)
		GPIO.output(IN3, GPIO.HIGH)
		time.sleep(0.01)	
	else:
		GPIO.output(IN1, GPIO.HIGH)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.LOW)
		GPIO.output(IN4, GPIO.LOW)
		time.sleep(0.07)
		GPIO.output(IN1, GPIO.HIGH)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.HIGH)
		GPIO.output(IN4, GPIO.LOW)
		time.sleep(0.01)	
	
net = cv2.dnn.readNet("yolov3-tiny_final.weights", "yolov3-tiny.cfg")
classNames = []
with open("obj.names", "r") as f:
    classNames = [line.strip() for line in f.readlines()]
layers_names = net.getLayerNames()
output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]



class myThread(threading.Thread):
	def __init__(self,name,img):
		threading.Thread.__init__(self)	
		self.name = name
		self.img = img
	def run(self):
		global pridictValue,Thread1IsFinish,state,Thread2IsFinish,cx2,tmp
		if self.name == 'Thread1':
			Thread1IsFinish = 0
			start = time.time()
			#print('Thread1 is running')
			img = self.img
			img = img[0:400,700:1280]
			blur = cv2.GaussianBlur(img,(7,7),0)
			frame = cv2.addWeighted(img,2,blur,-1,0)
			img = cv2.resize(img,(640,480))
			height,width,channels = img.shape
			blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
			net.setInput(blob)
			outputs = net.forward(output_layers)
			boxes = []
			confs = []
			class_ids = []
			for output in outputs:
				for detect in output:
					scores = detect[5:]
					class_id = np.argmax(scores)
					conf = scores[class_id]
					if conf > 0.7:
						center_x = int(detect[0] * width)
						center_y = int(detect[1] * height)
						w = int(detect[2] * width)
						h = int(detect[3] * height)
						x = int(center_x - w/2)
						y = int(center_y - h / 2)
						boxes.append([x, y, w, h])
						confs.append(float(conf))
						class_ids.append(class_id)
						

			if len(class_ids) != 0:
				pridictValue = classNames[class_ids[0]] 
			else:
				pridictValue = 'Nothing'
						
			if tmp == 0:
				tmp = 1
			else:
				tmp = 0
						
			Thread1IsFinish = 1
			end =time.time()
			
		else:
			Thread2IsFinish = 0
			img =self.img
			img = cv2.resize(img,(640,480))
			img_crop1 = img[340:360,0:640] #trên 
			img_crop2 = img[460:480,0:640] #dưới
			cx1 = 320
			imgHsv1 = cv2.cvtColor(img_crop1, cv2.COLOR_BGR2HSV)
			lower = np.array([0, 0, 0])
			upper = np.array([179, 255, 90])
			mask1 = cv2.inRange(imgHsv1,lower,upper)
			result1 = cv2.bitwise_and(img_crop1,img_crop1, mask = mask1)
			gray1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
			ret1,thresh1 = cv2.threshold(gray1,10,255,cv2.THRESH_BINARY)
			contours1,hierarchy1 = cv2.findContours(thresh1, 1, cv2.CHAIN_APPROX_NONE)
	
			if len(contours1) > 0:
				c1 = max(contours1, key=cv2.contourArea)
				M1 = cv2.moments(c1)
				if M1['m00']!=0:
					cx1 = int(M1['m10']/M1['m00'])
					cy1 = int(M1['m01']/M1['m00'])
				else:
					cx1 = 320 
					cy1 = 320

			
			imgHsv2 = cv2.cvtColor(img_crop2, cv2.COLOR_BGR2HSV)
			lower = np.array([0, 0, 0])
			upper = np.array([179, 255, 90])
			mask2 = cv2.inRange(imgHsv2,lower,upper)
			result2 = cv2.bitwise_and(img_crop2,img_crop2, mask = mask2)
			gray2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
			ret2,thresh2 = cv2.threshold(gray2,10,255,cv2.THRESH_BINARY)
			contours2,hierarchy2 = cv2.findContours(thresh2, 1, cv2.CHAIN_APPROX_NONE)
			if len(contours2) > 0:
				c2 = max(contours2, key=cv2.contourArea)
				M2 = cv2.moments(c2)
				if M2['m00']!=0:
					cx2 = int(M2['m10']/M2['m00'])
					cy2 = int(M2['m01']/M2['m00'])
				else:
					cx2 = 320 
					cy2 = 320

			Vdtx = cx1 - cx2
			Vdty = 120
			Vunitx = 5
			Vunity =0
			#print(cx2)
			angle = (180/3.14)*math.acos((Vdtx*Vunitx+Vdty*Vunity)/(math.sqrt(Vdtx**2+Vdty**2)*math.sqrt(Vunitx**2+Vunity**2)))
			state = 'Forward'
			if cx2 < 140: # 0-140
				if angle < 20:
					state = 'Forward'
				else:
					state = 'Left'
			elif cx2 < 260: #140-246
				if angle < 55:
					state = 'Forward'
				else:
					state = 'Left'
			elif cx2 <320: #260-320
				if angle <90:
					state ='Forward'
				else:
					state = 'Left'
			elif cx2 < 380: #320-380
				if angle < 90:
					state = 'Right'
				else:
					state ='Forward'
			elif cx2 < 500: #380-500
				if angle >125:
					state = 'Forward'
				else:
					state = 'Right'
			else:
				if angle >160:
					state = 'Forward'
				else:
					state = 'Right'
			#print('Thread2 is running with state = ',end='\t')
			#print(state)
			Thread2IsFinish = 1




def MyFilter(img):
	imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower = np.array([0, 0, 0])
	upper = np.array([179, 255, 90])
	mask = cv2.inRange(imgHsv,lower,upper)
	result = cv2.bitwise_and(img,img, mask = mask)
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
	return thresh





tmp2 = 0 
cap = cv2.VideoCapture(0)
frameWidth = 1280
frameHeight = 720
cap.set(3, frameWidth)
cap.set(4, frameHeight)
while True:
	start = time.time()	
	ret,img = cap.read()
	if not ret:
		continue
	img2 = img[0:400,740:1280]
	cv2.imshow('IMG',img2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 



	if Thread1IsFinish == 1:
		Thread1 = myThread('Thread1',img)
		Thread1.start()
	if Thread2IsFinish == 1:
		Thread2 = myThread('Thread2',img)
		Thread2.start()
	direction = state
		
	
	
	print('Sign = ',end='')
	print(pridictValue,end = '||')
	print('Direction = ',end='')
	print(state,end = '||')

	if tmp2 != tmp:
		for i in range(0,len(KiemTraSuXuatHien)-1,1):
			KiemTraSuXuatHien[len(KiemTraSuXuatHien)-1-i] = KiemTraSuXuatHien[len(KiemTraSuXuatHien)-2-i]
		KiemTraSuXuatHien[0] = pridictValue
	
	if 'Forward' in KiemTraSuXuatHien:
		recentSign = 'Forward'
	else:
		if pridictValue != 'Nothing':
			recentSign = pridictValue
	tmp2 = tmp

	print('recentSign = ',end='')
	print(recentSign,end = '||')
	
	

	# Check intersection :v
	imgForCheck = cv2.resize(img,(640,480))
	img_crop = imgForCheck[340:480,0:640]
	img_crop = MyFilter(img_crop)
	contours,hierarchy = cv2.findContours(img_crop,1,cv2.CHAIN_APPROX_NONE)
	checkInterSection = 0
	if KienTraVuaRe ==0:
		if len(contours)>0:
			maxArea = 0
			for i in range(0,len(contours),1):
				area = cv2.contourArea(contours[i])
				if area > maxArea:
					maxArea = area
			#print(maxArea)
			if maxArea > 20000:
				KienTraVuaRe = KienTraVuaRe + 1
				if recentSign =='Right':
					Forward()
					time.sleep(0.3)
					GPIO.output(IN1, GPIO.HIGH)
					GPIO.output(IN2, GPIO.LOW)
					GPIO.output(IN3, GPIO.LOW)
					GPIO.output(IN4, GPIO.LOW)
					time.sleep(delayTime)
					checkInterSection = 1
				elif recentSign == 'Left':
					Forward()
					time.sleep(0.3)
					GPIO.output(IN1, GPIO.LOW)
					GPIO.output(IN2, GPIO.LOW)
					GPIO.output(IN3, GPIO.HIGH)
					GPIO.output(IN4, GPIO.LOW)
					time.sleep(delayTime)
					checkInterSection = 2
				else:
					Forward()
					time.sleep(delayTime-0.5)
					checkInterSection = 3
	else:
		if KienTraVuaRe == 10:
			KienTraVuaRe = 0
		else:
			KienTraVuaRe = KienTraVuaRe + 1
			
	if checkInterSection != 0:
		
		if checkInterSection == 1:
			print('RIGHT',end = '||' )
		elif checkInterSection == 2:
			print('LEFT',end = '||')
		else:
			print('FORWARD',end = '||')		
		checkInterSection = 0		
		continue
	print('False',end = '||')
	print("                ",end = "")
	print(KiemTraSuXuatHien)
	if pridictValue == 'Stop':
		#Is top sign in frame now ?
		if recentSign == 'Stop':
			# If right
			doWhat = 'Stop'
		else:
			doWhat = direction 
			
	else:	
		doWhat = direction
	
	if doWhat == 'Right':
		Right(cx2)
	elif doWhat == 'Forward':
		Forward()
	elif doWhat == 'Left':
		Left(cx2)
	else:
		Stop()

	end =time.time()
	#print('Thoi Gian',end = " ")
	#print(end -start)



GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
