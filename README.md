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
