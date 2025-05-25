# Facial Expression Recognition Challenge

## Overview

This project tackles the Facial Expression Recognition Challenge from Kaggle, where the goal is to classify facial expressions from grayscale images into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The dataset contains 48×48 pixel grayscale facial images with corresponding emotion labels. This challenge tests the ability of machine learning models to recognize human emotions from visual data.

## Dataset

- Training set: 28,709 examples
- Public test set: 3,589 examples
- Private test set: 3,589 examples

## Evaluation

Models are evaluated based on their accuracy in predicting the correct emotion class for each image.

## Original Competition

https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview

# დავიწყოთ:

კაი ნუ წავიკითხე პირობა, 2013 ისე კაი ცოტა ფულს უხდიდნენ მოგებულს (300$) -თან კაი რეპორტებიც წერეთო მარა როგორც ჩანს არ იყო განვითარებული ეს სფერო. ოქეი, 48x48 pixel ესეთი სურათები გვაქ, (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) 7 კატეგორიით. 28,709 ამდენი განსხვავებული სახე გვაქ- ნუ არც თუ ისე დიდი დატასეტია მარა ალბათ არ დაგვჭირდება მეტი. ფოტოები რო არაა ცოტა ცუდია მარა ალბათ კომფიდენციალურობის თემაა და მაგიტო ( პიქსელები ჩანს მხოლოდ).

დავიწყოთ colab-ში მუშაობა. ა ხო ისე ძაან საინტერესო დავალებაა მომეწონა. ვნახოთ მუშაობის პროცესი როგორი იქნება.

ეს wandb და mlflow ებთან კაი პრობლემები მაქ ხოლმე მარა ვცადოთ აბა ეხა როგორ იქნება.

# წავიდა მუშაობა.

# EXPERIMENT 1
ფაილი: Facial_Expression_Recognition_1.ipynb

დავაყენე საჭირო რაღაცეები და გადმოვტვირთე kaggle dataset.
wandb- მგონი უკეთესად ყენდება ვიდრე mlflow მარტივად ქნა ყველაფერი ეხა უბრალოდ სატესტოდ ვნახოთ რაღაცეები და ცოტა მარტივი არქიტექტურა ავაგოთ.

ვაა პიქსელებში რო იყო იდეაში, ნახვა სახეების შესაძლებელი ყოფილა, pixels_to_image ფუნქციით სახეები ვნახე.

Training set size: 28709
Test set size: 7178
Distribution of emotions in training set:
Angry: 3995 (13.9%)
Disgust: 436 (1.5%)
Fear: 4097 (14.3%)
Happy: 7215 (25.1%)
Sad: 4830 (16.8%)
Surprise: 3171 (11.0%)
Neutral: 4965 (17.3%)

ნუ disgust სხვებთან შედარებით ბევრად ცოტაა, მარა სხვების განაწილება ნორმალურია.

FER2013Dataset ში პიქსელები numpy array-დ გადავაქციოთ.
ნუ დავამუშაოთ ისე რო მათი გამოყენება შესალებელი იყოს.
დასაწყისისთვის გავაკეთოთ ძალიან პატარა network.
48×48 = 2304 ჯერ დავაflatten-ოთ
მერე პირდაპირ სატესტოთ ძალიან შევამციროთ 2304 → 64 და მერე 64 → 7
ასევე relu-ს გამოვიყენება რომ linear არ იყოს.

საწყოსისთვის 10 ეპოქიანი დავატრეინინგოთ.

torch-ს გამოვიყენებ ნუ ჯერ კარგად ვერ ვერკვევი ამიტო ცოტა AI გამოვიყენოთ, მარა შემდეგი ექსპერიმენტებზე ნელ-ნელა დავამუღამებ.

Original data size: 28709
Training set size: 20670 (72.0%)
Validation set size: 5168 (18.0%)
Test set size: 2871 (10.0%) -ესე გავყავი data - თავიდან დამავიწყდა data-ს გაყოფა და ამიტო წავშალე კოდი და ეხა ახლიდან
მარა ყველაფერს იგივეს ვაკეთებ და ეხა ვაგრძელებ ტრეინინგიდან.
ნორმალიზაცია გავუკეთე პიქსელებს. batch size -64 გვექნება დასაწყისისთვის lr- 0.02 , epochs - 10.
 
დაიწყო ტრეინინგი ვნახოთ პირველ ცდაზე რა გვექნება შედეგი.

იდეაში მარტივი მოდელია მარა 5 წუთამდე დრო მაინც წაიღო.

Epoch 1/10, Train Loss: 2.2368, Train Acc: 26.19%, Val Loss: 2.3034, Val Acc: 29.18%
Epoch 2/10: 100%
 323/323 [00:27<00:00,  8.33it/s]
Epoch 2/10, Train Loss: 2.3093, Train Acc: 26.41%, Val Loss: 2.2460, Val Acc: 26.55%
Epoch 3/10: 100%
 323/323 [00:29<00:00, 13.59it/s]
Epoch 3/10, Train Loss: 2.3177, Train Acc: 27.40%, Val Loss: 2.4515, Val Acc: 25.56%
Epoch 4/10: 100%
 323/323 [00:27<00:00, 13.36it/s]
Epoch 4/10, Train Loss: 2.2902, Train Acc: 28.34%, Val Loss: 2.6144, Val Acc: 29.02%
Epoch 5/10: 100%
 323/323 [00:28<00:00, 13.60it/s]
Epoch 5/10, Train Loss: 2.2277, Train Acc: 27.87%, Val Loss: 2.0749, Val Acc: 27.59%
Epoch 6/10: 100%
 323/323 [00:27<00:00, 10.43it/s]
Epoch 6/10, Train Loss: 2.1335, Train Acc: 29.01%, Val Loss: 2.1371, Val Acc: 29.32%
Epoch 7/10: 100%
 323/323 [00:27<00:00,  8.21it/s]
Epoch 7/10, Train Loss: 2.1096, Train Acc: 28.78%, Val Loss: 2.1421, Val Acc: 28.42%
Epoch 8/10: 100%
 323/323 [00:28<00:00, 13.82it/s]
Epoch 8/10, Train Loss: 2.1046, Train Acc: 29.06%, Val Loss: 2.1739, Val Acc: 28.17%
Epoch 9/10: 100%
 323/323 [00:28<00:00, 13.47it/s]
Epoch 9/10, Train Loss: 2.0469, Train Acc: 30.12%, Val Loss: 2.0217, Val Acc: 29.18%
Epoch 10/10: 100%
 323/323 [00:28<00:00, 13.94it/s]
Epoch 10/10, Train Loss: 1.9689, Train Acc: 30.33%, Val Loss: 2.0559, Val Acc: 30.65%

ნუ ჩანს რომ ტრეინინგ სეტრის accuracy ნელ ნელა გაუმჯობესდა ასევე ვალიდაციის სეტიც. loss შემცირდა. ნუ ეს თავისთავად ცუდი შედეგია მარა პირველისთვის წავა ახლა ტესტ სეტი რო გამოვყავით მაგაზე ვნახოთ.

Test Loss: 1.9764, Test Accuracy: 30.30%

ჰმმ, ნუ იგივეა რაც ვალიდაციის სეტზე დაახლოებით. 

                precision    recall  f1-score   support

       Angry       0.16      0.05      0.08       399
     Disgust       0.00      0.00      0.00        44
        Fear       0.19      0.10      0.13       410
       Happy       0.49      0.49      0.49       722
         Sad       0.27      0.11      0.15       483
    Surprise       0.45      0.38      0.41       317
     Neutral       0.21      0.57      0.31       496

    accuracy                           0.30      2871
   macro avg       0.25      0.24      0.22      2871
weighted avg       0.30      0.30      0.28      2871

ბევრი რამის გამო გვაქ ესეთი შედეგი: ჰიპერპარამეტრები, არარი მოდელი კომპლექსური, შეიძლება სადღაც გრადიენტის გვიქრება მარა მაგას შემდეგ ექსპერიმენტებში გავტესტავ ეხა უბრალოდ მაინტერესებდა თუ მუშაობდა ამიტო აბდა უბდაა იდეაში.

Run summary:

epoch	10
test_accuracy	30.30303
test_loss	1.97643
train_accuracy	30.33382
train_loss	1.96885
val_accuracy	30.65015
val_loss	2.0559

https://wandb.ai/konstantine25b-free-university-of-tbilisi-/Facial_Expression_Recognition_1?nw=nwuserkonstantine25b

# EXPERIMENT 2
# კაი ეხა გავაუმჯობესოთ და სხვა ექსპერიმენტი დავიწყოთ:

ფაილი: Facial_Expression_Recognition_1.ipynb
