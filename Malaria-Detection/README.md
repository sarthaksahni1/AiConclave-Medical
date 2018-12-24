# Malaria-Detection-using-Keras

This project uses Keras to detect Malaria from Images. The model used is a ResNet50 which is trained from scratch.
The images in this [dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) is divided into to categories
- Parasitized
- Uninfected

## How to use


1. cd to the directory
```
cd Malaria-Detection
```

3. Get some images to infer upon
```
chmod u+x ./datasets/download.sh
./datasets/download.sh
```

4. Find an image of your choice and infer!!
```
python3 train_model.py -i ./path/to/image
```

5. Can even infer on a set of images in `datasets/cimages_test/`
```
python3 train_model.py -otb True
```

6. To see all the options
```
python3 train_model.py --help
```

## Production

The model is deployed to production and you can use the model to test on your own images!!<br/>
The model is deployed using [Zeit](https://zeit.co/).<br/>
The live link to the deployed model can be found here: https://malaria-classifier.now.sh <br/>

## A look into the deployed model on web

![malaria model - deployed classifier](https://user-images.githubusercontent.com/26242097/50305612-5d29eb80-04b9-11e9-9feb-7c0eb58483c6.png)

## deploy this models using this?

0. Download `node`, `now`, `now-cli`
```
sudo apt install npm
sudo npm install -g now
```

1. Get a **direct download** link to your model

2. Set that link equal to `model_file_url` - which you can find here on [app/server.py/L20](https://github.com/sarthaksahni1/AiConclave-Medical/blob/master/Malaria-Detection/zeit/app/server.py#L20)

3. Run
```
now
```
4. **The site should be deployed now!!**
