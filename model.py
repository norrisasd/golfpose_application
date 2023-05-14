from roboflow import Roboflow

rf = Roboflow(api_key="5egzYrifQQQMtUL0lseR")
project = rf.workspace().project("golfpose")
model = project.version(1).model
