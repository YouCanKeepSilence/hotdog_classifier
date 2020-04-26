# hotdog_classifier
Lab 1 of Deep Learning course in HSE. Need to classify images to two classes: hotdog and non-hotdog :)

Test accuracy (0.2 of full train data):
- CNN - 84.5% (20 epochs, 256 batch, 7 min to learn)
- SVM - 74% (20 min to learn)
- VGG16 + SVM - 95% (64 batch, pretrained Net, 10 secs to learn :))
- VGG16 - 60% (Too long to learn (> 40 min). Only 10 epochs were processed, 32 batch)

So I think that the most efficient approach to solve that task - VGG + SVM.