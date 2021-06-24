#Creating a dataset
import cv2
import numpy as np
import time

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        time.sleep(0.1)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

facelock_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
facelock_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

import smtplib
import os
import getpass
import imghdr
from os import environ 
from email.message import EmailMessage

def send_email():
    
    message = EmailMessage()
    
    
    #All variable values are set inside system's environmental variable
    
    
    SenderMail = environ.get('SENDER_MAIL')  #mail id of sender
    Recvmail = environ.get('RECEIVER_MAIL')    #password of sender
    SenderPass = environ.get('SENDER_PASS')#mail id of receiver 
    
    print("Sending mail....")
    
    message['subject'] = "This is text code by Balaji and his Team."
    message['from'] = SenderMail
    message['to'] = Recvmail
    message.set_content("Welcome to Face recognition app" )#it will only display if html content is not visible
    html_message = open("mail.html").read()
    message.add_alternative(html_message , subtype = "html")
    
    with open("./image/Crop.jpg" , "rb") as attach_file:
        image_name = attach_file.name
        image_type = imghdr.what(attach_file.name)
        image_data = attach_file.read()
        
    message.add_attachment(image_data , maintype = "image",
                          subtype = image_type , filename = "FaceLock.jpg"
                          )
    
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:#to connect with gmail server
        smtp.login(SenderMail,SenderPass)
        smtp.send_message(message)
        
    print("Email Sent sucessfully")
    
import pywhatkit              #pywhatkit library is used for whatsapp operation using python
from datetime import datetime #datetime module to get current time

def wpmsg():
    now = datetime.now()           # Get current time
    hr = int(now.strftime("%H"))   # Current Hour
    min = int( now.strftime("%M"))  # Current mint
    number = environ.get('MOB_NUMBER')
    pywhatkit.sendwhatmsg(number,"Hi Akshey , Someone tried to access your app", hr,min+1 ,wait_time=8)
    

def aws_cli_access():
    if FaceDetect == 70:
        unlock = cv2.imread("./image/unlock.jpg")
        
        cv2.putText(unlock, "Access", (45, 50) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(unlock, "Granted", (45, 100) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow("unlock" , unlock)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        print("=============================================")

        print("Launching aws ec2 instance")
        instance = os.system("aws ec2 run-instances  --image-id ami-0ad704c126371a549 --count 1 --instance-type t2.micro --tag-specifications  ResourceType=instance,Tags=[{Key=Name,Value=Task6}]"
)
        if instance == 0:
                print("EC2 instance launched!!!!")
        else:
            print("There's something error here! Instance not deployed")
            pass
            
            
        print("=============================================")
        print("Launching EBS volume")    
        ebs_vol = os.system("aws ec2 create-volume --volume-type gp2 --size 10 --availability-zone ap-south-1a  --tag-specifications  ResourceType=volume,Tags=[{Key=Name,Value=Task6EBS}]")  
        if ebs_vol == 0:
                print(" EBS volume launched")
        else:
            print("There's something error here! EBS not deployed")
            pass
        
        print("=============================================")    
        print("Saving instance id and volume id in variables")
        instance_id = subprocess.getoutput("aws ec2 describe-instances --filters Name=tag:Name,Values=Task6 --query Reservations[*].Instances[*].[InstanceId] --output text")
        vol_id = subprocess.getoutput("aws ec2 describe-volumes  --filters Name=tag:Name,Values=Task6EBS --query Volumes[*].[VolumeId] --output text")
        
        print("=============================================")
        print("Attaching EBS volume")      
        ebs_att = os.system("aws ec2 attach-volume --instance-id {0} --volume-id {1} --device /dev/sdf".format(instance_id,vol_id))
        if ebs_att == 0:
            print("EBS volume attached with EC2 sucessfully")
        else:
            print("There's something error here! Can't attach ebs to ec2")
            pass
        print("=============================================")
        print("Infrastructure deployed")
        print("=============================================")

        
        
    elif Anon == 120:
        lock = cv2.imread("./image/lock.jpg")
        cv2.putText(lock, "Access", (45, 50) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(lock, "Denied", (45, 100) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow("lock" , lock)
        cv2.waitKey(200)
        cv2.destroyAllWindows()
        
        face_anon = face_classifier.detectMultiScale(image, 1.3, 5)#This code will detect face from image 
        
        for x,y,w,h in face_anon:
            cimg1 = image[y:y+h, x:x+w,  : ]#Here it crops the image using ordinates of image 
                  
        #Below code was used to check output of cropped image
        #cv2.imshow("hii", cimg1)
        #cv2.waitKey(200)
        #cv2.destroyAllWindows()

        cv2.imwrite("./image/Crop.jpg",cimg1)#This will save our cropped picture inside /image folder
        print("=============================================")

        print("We are sending mail to owner of this account")
        send_email()
        print("=============================================")
        print("Sending whats app msg to owner")
        wpmsg()
        print("Message sent sucessfully to owner")
        print("=============================================")
        
        
import cv2
import numpy as np
import os
import subprocess




face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi



# Open Webcam
cap = cv2.VideoCapture(0)
FaceDetect = 0
Anon = 0

while True:

    ret, frame = cap.read()
    image, face = face_detector(frame)
    
    cv2.imwrite("./image/Mypic.jpg" , image )
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value

        results = facelock_model.predict(face)

        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence >= 90:
            cv2.putText(image, "Hii Pratik", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            FaceDetect += 1
            
        #elif confidence < 90:
        else:
            cv2.putText(image, "I dont know, who are you", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            Anon += 1
            
            
        #After matching with 70 frames it will break loop and go for next part where it will deploy infrastrure using terraform
        if FaceDetect == 70:
            print("Your AWS infrastructure will be deployed") 
            break
            
            
        #After fail to match with 70 frames it will break loop and go for next part where it will take pic of that user , crop it , send mail to owner with that pic and send whats app msg
        elif Anon == 120:
            print("Anon user")
            break
            
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        

cv2.destroyAllWindows()


# To call function where we wrote condition for sending mail+whats app message if face is not detected
#If it detects face , it will authenticate you and deploy terraform infrastructure
aws_cli_access() 

cap.release()        
