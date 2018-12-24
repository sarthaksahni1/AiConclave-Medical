# AiConclave-Medical
Our final solution consists of three Deep Learning models that offer accuracies of
over 90%, and can be quite reliably used for a quick check or a second opinion on
diseases. These models are hosted using a Flask-based API, for which these
models had to be turned into a service.
After installing the required libraries, one would just need to run the api.py file,
which would start the server on the local host. This would host three links, which,
after payment for the service, allow users to upload their image and give them an
accurate prediction of the disease.
Some of the diseases we would be tackling with this project are:
1.	Diagnosis of eye diseases using OCT imaging
2.	Malaria Detection using keras
3.	Skin Cancer (Melanoma) detection

## Eye Disease Detection
Optical coherence tomography (OCT) is an imaging technique that uses coherent light to capture high resolution images of biological tissues. OCT is heavily used by ophthalmologists to obtain high resolution images of the eye retina. Retina of the eye functions much more like a film in a camera. OCT images can be used to diagnose many retina related eyes diseases. Three eye diseases of  particular interest are listed below:<br>
    1.Choroidal neovascularization (CNV)<br>
    2.Macular Edema (DME)<br>
    3.Drusen (DRUSEN)<br><br>
Input Image:-<br>
![Eye Disease model - Input Image](http://blog.mapshalli.org/wp-content/uploads/2018/03/NORMAL-2038-3.jpeg)<br>
Output Image:-<br>
![Eye Disease model - Output Image](http://blog.mapshalli.org/wp-content/uploads/2018/03/drusen-heat.jpg)<br>

Add Custom Image to the model and check the Results:- <br>
Set that link equal to `predict_from_image_url` - which you can find here on [eyediseases-AI/Evaluate.ipynb/L40](https://github.com/sarthaksahni1/AiConclave-Medical/blob/master/eyediseases-AI/Evaluate.ipynb#L40)
<br>
## Malaria Detection
Live at: https://malaria-classifier.now.sh/
<br>
Here we are using Keras to detect Malaria from Images. The model used is a ResNet50 which is trained from scratch.<br>
The dataset contains 2 folders:<br>
 1.Infected<br>
 2.Uninfected<br>
And a total of 27,558 images.<br>
Output Interface:-
![malaria model](https://raw.githubusercontent.com/sarthaksahni1/AiConclave-Medical/master/images/Malaria%20Screen.PNG)

