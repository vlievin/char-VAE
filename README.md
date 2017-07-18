# Thesis Abstract

This thesis reviews the use of neural networks to build a general natural language model and evaluate its application to the task of linguistic style adaptation. We name style adaptation the process of transforming a sentence into another sentence which conveys the same meaning but uses a different linguistic style. This work has been strongly influenced by the algorithm for artistic style proposed in the computer vision field.

Hopefully, the work presented in this thesis will help to create better generative language model with near human performances. We believe that this technology could be a strong asset in the translation industry and could significantly improve current conversational interfaces.

The literature review has motivated the use of the Variational Auto-Encoder to build a continuous representation of the language. During this thesis, we have  rst conducted an in-depth study of the theory behind this framework and tested it against the case of the MNIST dataset modeling.

Motivated by the sequential natural of the language, we introduced the Recurrent Neural Network architecture and proven that it was compatible with the Variational Auto-Encoder framework by testing it against the modeling of a narrow set of time series.

As a second step, we applied the Variational Auto-Encoder to build a character-level language model. For this purpose, a narrow set of sentences taken from the Large Movie Review Dataset has been used. We reported poor generative performances but good recognition performances has been notably shown by its ability to rephrase unknown sentences.

Thereafter, we justified the choice of the sentiment as a specific case of stylistic feature and we proposed four different approaches for the task of style adaptation. Afterwards, we exploited the good recognition performances of our model to build a simple prototype which can be regarded as a successful case of style adaptation: the prototype has proven to be able to change the sentiment conveyed by simples sentences while addressing the same object.

We hope that the analysis of the Variational Auto-Encoder and its application to the task of linguistic style adaptation will motivate further research in this domain and we are convinced that our methods can be applied to the task of natural language understanding in the near future. Lastly, we propose further research in order to overcome the poor generative performances and apply this model to generative tasks.

# What is included

* report
* code to download and pre-process the Large Movie Review Dataset
* code for the VAE model used as language model
* code for the Pessimistic Machine

