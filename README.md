# Attacking-Neural-Networks
Testing DNNs is not a trivial task, as research highlights the many glaring susceptibilities of DNNs to adversarial attacks in image classification applications. These attacks entail applying a small, visually imperceptible perturbation to the input image such that the neural network outputs a misclassification. Such attacks can be catastrophic in safety-critical situations such as autonomous vehicles, aviation, medical technology, defense systems, etc. 

This repository contains scripts for adversarial attacks conducted on the following deep neural networks trained on the ImageNet dataset:
- ResNet50 (97.8MB)
- InceptionV3 (104 MB)
- VGG16 (528 MB)
- VGG19 (548 MB)

The following attacks have been performed using both single images and batch of images:
- Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
- Fast Gradient Sign Method (FGSM)
- DeepFool
- Carlini-Wagner Attack

We also focused on attacking a robust model trained on the CIFAR10 data and the paper outlining the model can be found here:
- **Towards Deep Learning Models Resistant to Adversarial Attacks**, *Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu* <br>
- Paper link: https://arxiv.org/abs/1706.06083.



