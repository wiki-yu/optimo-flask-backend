import cv2
def function(inputs,options):
		# print("Image canny runs here")
		return {"image": cv2.Canny(inputs["image"],100,200)}