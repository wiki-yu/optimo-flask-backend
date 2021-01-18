import cv2
def function(inputs,options):
		# print("Image gray runs here")
		return {"image": cv2.cvtColor(inputs["image"], cv2.COLOR_BGR2GRAY)}