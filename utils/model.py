from roboflow import Roboflow

def model():
    rf = Roboflow(api_key="MG3jnMckWXD6AP3z3ifM")
    project = rf.workspace().project("license-plate-recognition-rxg4e")
    model = project.version(4).model
    return model