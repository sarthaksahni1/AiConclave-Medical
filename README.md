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
![Eye Disease model - Input Image](https://raw.githubusercontent.com/sarthaksahni1/AiConclave-Medical/master/images/NORMAL-2038-3.jpeg)<br>
Output Image:-<br>
![Eye Disease model - Output Image](https://raw.githubusercontent.com/sarthaksahni1/AiConclave-Medical/master/images/index.png)<br>

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
Output Interface:-<br>
![malaria model](https://raw.githubusercontent.com/sarthaksahni1/AiConclave-Medical/master/images/Malaria%20Screen.PNG)
Paratized Cell Image
![malaria model paratized](https://storage.googleapis.com/kagglesdsdata/datasets%2F87153%2F200743%2Fcell_images%2Fcell_images%2FParasitized%2FC100P61ThinF_IMG_20150918_144104_cell_163.png?GoogleAccessId=datasets-dataviewer@kaggle-161607.iam.gserviceaccount.com&Expires=1545725934&Signature=GdyL85KBDVvcXBDhYLCdSq84CPAW0Ow0tkB%2Fy0w429iwTP6bE7CdZEut6QBjOS57vr1F88rVSYbH9X19PB9fAWjCg99J4%2BEiKMabLTkYlDhKkOqAwBnDzJPnZjr6mmQ21Tt%2B1DUBZxXc%2BaQJ9SyfjQPjgyCNOXgAq1CMrOmhHuyFQ3PoJgNHUp%2FnvrGQBrH2r78hsxIrkuItW3W1jRCGA5qOMXiE4EFlYZGVrEH1r4u7b6ZgOzGerudj0eBlLWQtdeWc4CeLq9daVt6U80rBDUNVoKJED6rTlJHSxLmU4qRMNxUGNmXTqLaJ%2BvVlzkOgal2%2BaF1885Pz2klp%2FqnaYA%3D%3D)
Uninfected Cell Image
![malaria model](https://storage.googleapis.com/kagglesdsdata/datasets%2F87153%2F200743%2Fcell_images%2Fcell_images%2FUninfected%2FC100P61ThinF_IMG_20150918_144104_cell_131.png?GoogleAccessId=datasets-dataviewer@kaggle-161607.iam.gserviceaccount.com&Expires=1545719681&Signature=eEsY0PEoKz%2Fn%2BVJpmksRbREELtdXsMwIWQS6WbxDfLVbVxunla4HmoRgcmsS3ZP3VHavwXM13nWy%2BG0fsVRDR2NIwtGISgm5mIOan6gkrnaWLGW1IJONHuguBAnLTgayX1S7szp%2F9LJxk5oHKixNNE2enqQuhgXejMTVtByDFYeJbKsRDF1LMX1DpbmyzLn9%2FvXEyQyOZrGCWduCLSivbarTJnTB5PIzXvYKvvZFRgzb5q2z03w%2FgigOT%2FF4%2BAE%2BGIsTRckKlTqF%2FTu%2B7FNw0LvZOy6E%2B6D%2FAXnLnmy6bDWUuW3qtJejK%2FuZTJh0pFiYRODp9KML7UujdB6eysao4Q%3D%3D)
<br>
## Skin Cancer Detection
