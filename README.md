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

macro avg 0.25 0.24 0.22 2871
weighted avg 0.30 0.30 0.28 2871

ბევრი რამის გამო გვაქ ესეთი შედეგი: ჰიპერპარამეტრები, არარი მოდელი კომპლექსური, შეიძლება სადღაც გრადიენტის გვიქრება მარა მაგას შემდეგ ექსპერიმენტებში გავტესტავ ეხა უბრალოდ მაინტერესებდა თუ მუშაობდა ამიტო აბდა უბდაა იდეაში.

Run summary:

epoch 10
test_accuracy 30.30303
test_loss 1.97643
train_accuracy 30.33382
train_loss 1.96885
val_accuracy 30.65015
val_loss 2.0559

https://wandb.ai/konstantine25b-free-university-of-tbilisi-/Facial_Expression_Recognition_1?nw=nwuserkonstantine25b

მოკლედ როცა github-ზე გადატანა ვცადე ამ ერორს მიგდებდა:
Invalid Notebook
There was an error rendering your Notebook: the 'state' key is missing from 'metadata.widgets'. Add 'state' to each, or remove 'metadata.widgets'.
Using nbformat v5.10.4 and nbconvert v7.16.6
ნუ ახლიდანაც გავუშვი, სემდეგ გადმოვწერე ნოუთბუქი და ისე ავტვირთე მარა სანამ ყველა output არ წავშალე მანამდე არ მოეხსნა ეს ერორი. ამიტომ githubze ეს ipynb ფაილი output-ების გარეშეა.

# EXPERIMENT 2

# კაი ეხა გავაუმჯობესოთ და სხვა ექსპერიმენტი დავიწყოთ:

ფაილი: Facial_Expression_Recognition_2.ipynb

მოკლედ ახლა უკვე გავართულოთ მოდელი, learning rate იყოს 0.001,
batch 32, epochs 20, ასევე შემოვიღოთ dropout როგორც ლექციებზე ვქენით და იყოს 0.3. და ასევე weight decay-ც შემოვიღოთ.
იგივენაირად დატა ლოადინგი, flattening, ნორმალიზაცია.

ახლა ავდგეთ და ვნახოთ თუ მოხდება overfit როცა 20 ცალუ გვაქ.
მაგრამ ჯერ ავაგოთ მარტივი CNN. 
გამოვიყენოთ BatchNormalization and max pooling და dropout.

ნუ adamს და crossentropyloss-ს ვიყენებთ როგორც წესია.

ამასტან ერთად დავამატე ReduceLROnPlateau რომელიც როცა მოდელი წვალობს გაუსწორებს lr-ს და დაეხმარება სწავლაში.

გავუშვი მოდელი 20 ცალ დატაზე და ვნახოთ თუ წავა ოვერფიტში:
Overfit Epoch 1/30, Loss: 1.9313, Acc: 30.00%
Overfit Epoch 2/30, Loss: 2.2543, Acc: 55.00%
Overfit Epoch 3/30, Loss: 1.3586, Acc: 50.00%
Overfit Epoch 4/30, Loss: 1.6047, Acc: 55.00%
Overfit Epoch 5/30, Loss: 1.4360, Acc: 50.00%
Overfit Epoch 6/30, Loss: 0.5932, Acc: 80.00%
Overfit Epoch 7/30, Loss: 0.5074, Acc: 70.00%
Overfit Epoch 8/30, Loss: 0.7377, Acc: 75.00%
Overfit Epoch 9/30, Loss: 0.3926, Acc: 90.00%
Overfit Epoch 10/30, Loss: 0.2310, Acc: 85.00%
Overfit Epoch 11/30, Loss: 0.1551, Acc: 95.00%
Overfit Epoch 12/30, Loss: 0.0240, Acc: 100.00%
Overfit Epoch 13/30, Loss: 0.1456, Acc: 90.00%
Overfit Epoch 14/30, Loss: 0.0368, Acc: 100.00%
Overfit Epoch 15/30, Loss: 0.0266, Acc: 100.00%
Overfit Epoch 16/30, Loss: 0.0316, Acc: 100.00%
Overfit Epoch 17/30, Loss: 0.0530, Acc: 100.00%
Overfit Epoch 18/30, Loss: 0.0068, Acc: 100.00%
Overfit Epoch 19/30, Loss: 0.0046, Acc: 100.00%
Overfit Epoch 20/30, Loss: 0.0081, Acc: 100.00%
Overfit Epoch 21/30, Loss: 0.0096, Acc: 100.00%
Overfit Epoch 22/30, Loss: 0.0051, Acc: 100.00%
Overfit Epoch 23/30, Loss: 0.0023, Acc: 100.00%
Overfit Epoch 24/30, Loss: 0.0025, Acc: 100.00%
Overfit Epoch 25/30, Loss: 0.0069, Acc: 100.00%
Overfit Epoch 26/30, Loss: 0.0761, Acc: 95.00%
Overfit Epoch 27/30, Loss: 0.0237, Acc: 100.00%
Overfit Epoch 28/30, Loss: 0.0058, Acc: 100.00%
Overfit Epoch 29/30, Loss: 0.0453, Acc: 100.00%
Overfit Epoch 30/30, Loss: 0.0006, Acc: 100.00%

კი წავიდა ოვერფიტში ამიტომ ნაკლები შანსია vanishing gradient-ის

ახლა კიდე გავუშვი დასატრეინინგებლად და ალბათ კაი 1 საათი ან მეტი დაჭირდება

გათიშვა მომიწია რადგან შევამჩნიე რომ cpu-ს იყენებდა da cuda-ზე გადართვა დამავიწყდა ამიტომ ალბათ ბევრად ნაკლებ დროში იზავს ახლიდან გავუშვი ეხა.
აქაც 20 ცალი ისევ ოვერფიტში წავიდა.
კიდე კაი cuda ჩავრთე ალბათ 15 წუთში დაასრულებს ტრეინინგს.

Epoch 16/20, Train Loss: 1.1178, Train Acc: 55.14%, Val Loss: 1.3190, Val Acc: 52.24%
New best model saved with validation accuracy: 52.24%
Epoch 17/20: 100%
 646/646 [00:21<00:00, 35.17it/s, loss=1.06, acc=56.9]
Epoch 17/20, Train Loss: 1.0781, Train Acc: 56.88%, Val Loss: 1.3218, Val Acc: 53.31%
New best model saved with validation accuracy: 53.31%
Epoch 18/20: 100%
 646/646 [00:22<00:00, 26.34it/s, loss=1.17, acc=58.8]
Epoch 18/20, Train Loss: 1.0235, Train Acc: 58.79%, Val Loss: 1.3228, Val Acc: 53.70%
New best model saved with validation accuracy: 53.70%
Epoch 19/20: 100%
 646/646 [00:20<00:00, 25.43it/s, loss=1.06, acc=59.8]
Epoch 19/20, Train Loss: 0.9983, Train Acc: 59.83%, Val Loss: 1.3236, Val Acc: 52.84%
Epoch 20/20: 100%
 646/646 [00:20<00:00, 39.83it/s, loss=1.06, acc=60.9]
Epoch 20/20, Train Loss: 0.9673, Train Acc: 60.90%, Val Loss: 1.3468, Val Acc: 53.93%
New best model saved with validation accuracy: 53.93%

Test Accuracy: 54.37%

ხო ეს წინა მოდელზე ბევრად უკეთესია.

                 precision    recall  f1-score   support

       Angry       0.46      0.41      0.43       399
     Disgust       0.00      0.00      0.00        44
        Fear       0.38      0.32      0.35       410
       Happy       0.76      0.77      0.76       722
         Sad       0.39      0.57      0.47       483
    Surprise       0.71      0.63      0.67       317
     Neutral       0.51      0.48      0.49       496

    accuracy                           0.54      2871
   macro avg       0.46      0.45      0.45      2871
weighted avg       0.54      0.54      0.54      2871

# გითჰუბზე კიდევ იგივე ერორი ქონდა მაგრამ როცა დავკლონე იმუშავა ჩვეულებრივად და ყველაფერი მიჩანს, როგორც ჩანს უბრალოდ გითჰუბს არ აქ სუფორთი რო აჩვენოს vs code-ში colab-ში მიღებული output-ები ჩვეულებრივად ჩანს. 
მაგრამ მოდი მაინც ავტვირთავ output-ების გარეშე რო თუ გადმოწერის დრო არ გექნებათ პირდაპირ შეხედოთ რო კოდი სწორად წერია :)


https://wandb.ai/konstantine25b-free-university-of-tbilisi-/Facial_Expression_Recognition_2?nw=nwuserkonstantine25b

# Experiment 3

ფაილი: Facial_Expression_Recognition_3.ipynb

კიდე უკეთესი შეგვიძლია ალბათ კომპლექსურობას თუ გავუზრდით მოდელს და სხვადასხვა ახალ რაღაცეებს შევხედავთ.

ამ შემთხვევაშიც დატა იგივე ნაირად მოგვაქ, დაკავშირებაც იგივე ნაირად. უბრალოდ ცოტა სხვა მიდგომებს და სხვა ჰიპერპარამეტრებით გავტესტავ.

აქ უკვე გვინდა სხვა ბევრი ჰიპერპარამეტრის გატესტვა ამიტო Sweet config გვაქ.

დავამატე ესენი 
transforms.RandomHorizontalFlip(), - ამიტ ვფლიპავთ ფოტოს და ეს გვეხმარება რომ მარცხნიდავ და მარჯვნიდან ორივე ნაირად შევხედოთ.
transforms.RandomRotation(10), ამითი rotations ვუკეთებთ და ესეც ერთგვარად გვეხმარება.


რაც შეეხებე თვითონ მოდელს ეს ბევრად უფრო კომპლექსურია წინასთან შედარებით.  გვაქვს Fully Connected Layer და 3 cnn.

ასევე BatchNorm, dropout და maxPooling.

გავტესტოთ 20 ცალზე:
Overfit Epoch 1/30, Loss: 2.4100, Acc: 15.00%
Overfit Epoch 2/30, Loss: 1.3525, Acc: 40.00%
Overfit Epoch 3/30, Loss: 1.4272, Acc: 60.00%
Overfit Epoch 4/30, Loss: 1.0353, Acc: 85.00%
Overfit Epoch 5/30, Loss: 1.1916, Acc: 80.00%
Overfit Epoch 6/30, Loss: 0.9157, Acc: 85.00%
Overfit Epoch 7/30, Loss: 1.0702, Acc: 80.00%
Overfit Epoch 8/30, Loss: 0.9121, Acc: 80.00%
Overfit Epoch 9/30, Loss: 0.7126, Acc: 90.00%
Overfit Epoch 10/30, Loss: 0.6071, Acc: 90.00%
Overfit Epoch 11/30, Loss: 0.7348, Acc: 90.00%
Overfit Epoch 12/30, Loss: 0.6527, Acc: 90.00%
Overfit Epoch 13/30, Loss: 0.4649, Acc: 100.00%
Overfit Epoch 14/30, Loss: 0.8879, Acc: 85.00%
Overfit Epoch 15/30, Loss: 0.5619, Acc: 95.00%
Overfit Epoch 16/30, Loss: 0.7871, Acc: 90.00%
Overfit Epoch 17/30, Loss: 0.3112, Acc: 100.00%
Overfit Epoch 18/30, Loss: 0.5016, Acc: 100.00%
Overfit Epoch 19/30, Loss: 0.3922, Acc: 90.00%
Overfit Epoch 20/30, Loss: 0.3409, Acc: 100.00%
Overfit Epoch 21/30, Loss: 0.4077, Acc: 100.00%
Overfit Epoch 22/30, Loss: 0.4692, Acc: 95.00%
Overfit Epoch 23/30, Loss: 0.4481, Acc: 95.00%
Overfit Epoch 24/30, Loss: 0.4839, Acc: 95.00%
Overfit Epoch 25/30, Loss: 0.4620, Acc: 95.00%
Overfit Epoch 26/30, Loss: 0.3034, Acc: 100.00%
Overfit Epoch 27/30, Loss: 0.2267, Acc: 100.00%
Overfit Epoch 28/30, Loss: 0.3239, Acc: 100.00%
Overfit Epoch 29/30, Loss: 0.2927, Acc: 100.00%
Overfit Epoch 30/30, Loss: 0.2521, Acc: 100.00%

აქაც ჩააბარა გამოცდა და ოვერფიტშია.

ჯერ გავუშვებ default პარამეტრებით 
default_config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'dropout_rate': 0.3,
    'weight_decay': 1e-5,
    'hidden_dim': 128
}
შემდეგ უკვე პარამატრებს გავტესტავ.

default- პარამეტრებს ალბათ დაჭირდება 20 წუთი, ვნახოთ.


Epoch 20/30, Train Loss: 1.0496, Train Acc: 60.04%, Val Loss: 1.1220, Val Acc: 57.95%
Epoch 21/30 [Train]: 100%
 323/323 [00:20<00:00, 18.34it/s]
Epoch 21/30, Train Loss: 1.0501, Train Acc: 60.30%, Val Loss: 1.1130, Val Acc: 58.44%
Epoch 22/30 [Train]: 100%
 323/323 [00:20<00:00, 18.48it/s]
Epoch 22/30, Train Loss: 0.9931, Train Acc: 62.84%, Val Loss: 1.0611, Val Acc: 60.20%
Model improved! Saved checkpoint (Val Acc: 60.20%)
Epoch 23/30 [Train]: 100%
 323/323 [00:20<00:00, 18.35it/s]
Epoch 23/30, Train Loss: 0.9644, Train Acc: 63.77%, Val Loss: 1.0608, Val Acc: 60.86%
Model improved! Saved checkpoint (Val Acc: 60.86%)
Epoch 24/30 [Train]: 100%
 323/323 [00:22<00:00, 17.54it/s]
Epoch 24/30, Train Loss: 0.9536, Train Acc: 64.08%, Val Loss: 1.0762, Val Acc: 59.98%
Epoch 25/30 [Train]: 100%
 323/323 [00:23<00:00, 12.45it/s]
Epoch 25/30, Train Loss: 0.9343, Train Acc: 64.80%, Val Loss: 1.0759, Val Acc: 60.14%
Epoch 26/30 [Train]: 100%
 323/323 [00:40<00:00, 17.54it/s]
Epoch 26/30, Train Loss: 0.9253, Train Acc: 65.14%, Val Loss: 1.0913, Val Acc: 59.27%
Epoch 27/30 [Train]: 100%
 323/323 [00:20<00:00, 18.21it/s]
Epoch 27/30, Train Loss: 0.9224, Train Acc: 65.14%, Val Loss: 1.0834, Val Acc: 59.73%
Epoch 28/30 [Train]: 100%
 323/323 [00:21<00:00, 20.77it/s]
Epoch 28/30, Train Loss: 0.8668, Train Acc: 67.33%, Val Loss: 1.0783, Val Acc: 60.14%
Epoch 29/30 [Train]: 100%
 323/323 [00:22<00:00, 17.79it/s]
Epoch 29/30, Train Loss: 0.8481, Train Acc: 68.48%, Val Loss: 1.0876, Val Acc: 60.47%
Epoch 30/30 [Train]: 100%
 323/323 [00:23<00:00, 12.85it/s]
Epoch 30/30, Train Loss: 0.8382, Train Acc: 68.72%, Val Loss: 1.0855, Val Acc: 60.49%
Final Test Accuracy: 60.64%


ანუ ვალიდაცია 60 % მდე ავიყვანეთ უკეთესია.

ასევე რო შევხედოთ ცხრილს თავიდან confusion matrix-ში არ იყო საერთოდ disgust- მაგრამ თანდათან დაემატა.

კაი ეხა გავუშვი 
print("\nRunning hyperparameter sweep to find the best model...")
wandb.agent(sweep_id, train_model, count=3)

რაც ადგება და ჰიპერპარამეტრებს გატესტავს

პირველი ვარიანტი:
wandb: 	batch_size: 128
wandb: 	dropout_rate: 0.2
wandb: 	hidden_dim: 64
wandb: 	learning_rate: 0.001
wandb: 	weight_decay: 1e-06

აქ ცოტა დიდი Batch გვაქ, 
dropout ნორმალური ზომისაა, lr დაბალია მარა ნორმალურია მაინც.
რაც შეეხება: weight_decay - ეს მცირე რეგულარიზაციაა. 

 Epoch 18/20, Train Loss: 0.8629, Train Acc: 67.25%, Val Loss: 1.1197, Val Acc: 60.20%
Model improved! Saved checkpoint (Val Acc: 60.20%)
Epoch 19/20 [Train]: 100%
 162/162 [00:20<00:00,  9.26it/s]
Epoch 19/20, Train Loss: 0.8508, Train Acc: 67.94%, Val Loss: 1.1244, Val Acc: 60.12%
Epoch 20/20 [Train]: 100%
 162/162 [00:21<00:00,  9.62it/s]
Epoch 20/20, Train Loss: 0.8234, Train Acc: 69.19%, Val Loss: 1.1234, Val Acc: 60.45%
Model improved! Saved checkpoint (Val Acc: 60.45%)
Final Test Accuracy: 60.61%

მეორე ვარიანტი: 
wandb: 	batch_size: 32
wandb: 	dropout_rate: 0.3
wandb: 	hidden_dim: 128
wandb: 	learning_rate: 0.01
wandb: 	weight_decay: 1e-06

აქ მცირე batch_size, dropout ნორმალურია, hidden_dim საშუალოზე ოდნავ დიდია. learning_rate მაღალია ვფიქრობ უფრო პატარა უკეთესი იქნება მარა მაინც გავტესტოთ.

Epoch 18/20, Train Loss: 0.9889, Train Acc: 62.61%, Val Loss: 1.1044, Val Acc: 59.33%
Epoch 19/20 [Train]: 100%
 646/646 [00:23<00:00, 27.80it/s]
Epoch 19/20, Train Loss: 0.9655, Train Acc: 63.59%, Val Loss: 1.1155, Val Acc: 60.06%
Model improved! Saved checkpoint (Val Acc: 60.06%)
Epoch 20/20 [Train]: 100%
 646/646 [00:24<00:00, 32.47it/s]
Epoch 20/20, Train Loss: 0.9518, Train Acc: 63.59%, Val Loss: 1.1270, Val Acc: 59.42%
Final Test Accuracy: 60.61%

მესამე ვარიანტი: 

wandb: 	batch_size: 32
wandb: 	dropout_rate: 0.2
wandb: 	hidden_dim: 256
wandb: 	learning_rate: 0.005
wandb: 	weight_decay: 0.0001

აქ მცირე batch_size, dropout ნორმალურია, hidden_dim საშუალოზე დიდია. learning_rate მაღალია ვფიქრობ რომ ნორმალურია. decay ნორმალურია.


Epoch 18/20, Train Loss: 1.0205, Train Acc: 61.46%, Val Loss: 1.0957, Val Acc: 59.67%
Epoch 19/20 [Train]: 100%
 646/646 [00:24<00:00, 20.69it/s]
Epoch 19/20, Train Loss: 1.0042, Train Acc: 61.90%, Val Loss: 1.0868, Val Acc: 59.89%
Epoch 20/20 [Train]: 100%
 646/646 [00:24<00:00, 22.76it/s]
Epoch 20/20, Train Loss: 0.9929, Train Acc: 62.31%, Val Loss: 1.0840, Val Acc: 59.42%
Final Test Accuracy: 60.99%

ხოლო ბოლოს საბოლოო ამ მოდელის ტესტინგი:

Final Test Accuracy with best model: 60.61%

Classification Report:
                 precision    recall  f1-score   support

       Angry       0.52      0.51      0.51       399
     Disgust       0.70      0.32      0.44        44
        Fear       0.46      0.39      0.42       410
       Happy       0.83      0.80      0.81       722
         Sad       0.45      0.61      0.52       483
    Surprise       0.71      0.71      0.71       317
     Neutral       0.60      0.53      0.56       496

    accuracy                           0.61      2871
   macro avg       0.61      0.55      0.57      2871
weighted avg       0.61      0.61      0.61      2871

Best run found: scarlet-sweep-1
Best validation accuracy: 60.45%
ამ შემთხვევაში მიღეთ რომ ეს არის საუკეთესო ჰიპერპარამეტრებიანი მოდელი.
Hyperparameters: {'batch_size': 128, 'hidden_dim': 64, 'dropout_rate': 0.2, 'weight_decay': 1e-06, 'learning_rate': 0.001}

ანუ აქ უკვე 60%+ ზე ვართ ასულები.

აქვე ანალოგიურად როგორც მეორეში და პირველში გითჰუბზე არ ჩანს გამოვიკვლიე და პრობლემა აქვს ტრეინინგს რომ ვუშვებ მანდ ჩემი აზრით თითოეული ეპოქის პრეოგრეს bar ებს ვერ აღიქვამს გითჰუბი. ნუ ეგ ვერ დავფიქსე ამიტო თუ ნახვა გინდათ უშუალოდჩამოტვირთეთ ფაილი ან რეფო და vs code-ში გამოჩნდება outputebi. output-ების გარეშე ავტვირთავ დამატებით.

