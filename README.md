# Image Captioning with Deep Learning

## Abstract

This project presents an advanced image captioning system leveraging a combination of CNN and RNN architectures. The system employs the VGG16 model for robust image feature extraction and an LSTM-based sequence model for generating descriptive captions. The approach involves intricate preprocessing of both visual and textual data, efficient model training with a custom data generator, and effective evaluation using BLEU scores. This end-to-end system demonstrates the synergy between deep learning techniques in computer vision and natural language processing to create meaningful, contextually relevant image captions.

## Overview

This project implements an image captioning system using a deep learning approach. It integrates a convolutional neural network (CNN) for feature extraction from images and a recurrent neural network (RNN) for generating captions. The CNN employed is VGG16, a pre-trained model, and the RNN consists of LSTM (Long Short-Term Memory) units.

### Image Feature Extraction

1. **Model Loading**: The VGG16 model, pre-trained on ImageNet, is loaded without its top classification layer, leaving the convolutional base for feature extraction.

    ```python
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    ```

2. **Feature Extraction Process**: Each image is loaded, preprocessed to match VGG16's input requirements, and passed through the model to obtain a feature vector.

    ```python
    for img_name in tqdm(os.listdir(directory)):
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[image_id] = feature
    ```

### Text Data Processing

1. **Caption Loading**: Captions are loaded from a text file, and a mapping from image IDs to captions is created.

    ```python
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        image_id, caption = tokens[0], tokens[1:]
        mapping[image_id].append(caption)
    ```

2. **Text Preprocessing**: Each caption is cleaned by converting to lowercase, removing non-alphabetic characters, and adding start and end sequence tokens.

    ```python
    def clean(mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i].lower().replace('[^A-Za-z]', '')
                caption = 'startseq ' + caption + ' endseq'
                captions[i] = caption
    ```

3. **Tokenization**: A tokenizer is created and fit on all the captions to create a vocabulary.

    ```python
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    ```

### Model Building

1. **Feature Extractor Layer**: A dense layer is added on top of the VGG16 features for dimensionality reduction and normalization.

    ```python
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    ```

2. **Sequence Processor**: An embedding layer and LSTM layers process the input sequence (captions).

    ```python
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
    se2 = LSTM(512, return_sequences=True)(se1)
    se3 = LSTM(512)(se2)
    ```

3. **Decoder**: The outputs of the feature extractor and sequence processor are merged and passed through dense layers to produce the final output.

    ```python
    decoder1 = add([fe2, se3])
    outputs = Dense(vocab_size, activation='softmax')(decoder1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    ```

### Training

The model is trained using a custom data generator that yields batches of image features, input sequences, and target words. Early stopping and model checkpointing are used to avoid overfitting and save the best model.

### Inference

For generating captions, a function predicts the next word in the sequence iteratively until the end sequence token is generated or the maximum length is reached.

```python
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        word = idx_to_word(yhat, tokenizer)
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text
```

### Evaluation

The model's performance is evaluated using BLEU scores, which measure the quality of the generated captions compared to the actual captions.

| Term                   | Definition                                                                                                                                                                                                                              | Application in Project                                                                                                                                                                     |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Convolutional Neural Network (CNN)** | A type of deep neural network used primarily in processing data with a grid-like topology, such as images.                                                                      | The VGG16, a pre-trained CNN, is used to extract high-level features from images.                                                                                                          |
| **VGG16**              | A CNN model known for its simplicity and depth, consisting of 16 convolutional layers. It's widely used for image classification tasks.                                         | Employed as a feature extractor for images.                                                                                                                                                |
| **Recurrent Neural Network (RNN)** | A class of neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior.                 | Used to process the sequences of words in the captions.                                                                                                                                    |
| **Long Short-Term Memory (LSTM)** | A special kind of RNN, capable of learning long-term dependencies. LSTMs are designed to avoid the long-term dependency problem, remembering information for extended periods. | Utilized in the RNN to remember the context in the captions.                                                                                                                               |
| **Feature Extraction** | The process of reducing the amount of resources required to describe a large set of data accurately.                                                                            | Extracting important features from images using VGG16.                                                                                                                                     |
| **Tokenization**       | The process of converting text into tokens, which are small units like words or phrases.                                                                                       | Captions are tokenized into individual words to create a vocabulary for the text data.                                                                                                     |
| **Embedding Layer**    | A layer in a neural network that turns positive integers (indexes) into dense vectors of fixed size. Typically used for processing text.                                        | Converts tokenized words into vectors for the LSTM to process.                                                                                                                             |
| **Dropout**            | A regularization technique where randomly selected neurons are ignored during training, which helps in preventing overfitting.                                                   | Applied in the CNN and RNN layers to improve generalization.                                                                                                                               |
| **Dense Layer**        | A fully connected neural network layer where each input node is connected to each output node.                                                                                 | Used after feature extraction and sequence processing to make final predictions.                                                                                                           |
| **Activation Function**| A function applied to the output of a neural network layer, which introduces non-linear properties to the model.                                                                | The 'relu' and 'softmax' activation functions are used.                                                                                                                                   |
| **Early Stopping**     | A form of regularization used to avoid overfitting by halting the training process if the model's performance ceases to improve on a hold-out validation dataset.                | Used to stop training the model when the validation loss stops improving.                                                                                                                  |
| **Model Checkpointing**| The practice of saving a model at intermittent points during training, particularly when the model performs better than in previous epochs.                                     | Saves the best model based on validation loss.                                                                                                                                             |
| **Preprocessing**      | The technique of performing operations on the data before feeding it into the model. This can include scaling, normalizing, or transforming the data.                           | Images are preprocessed to match the input requirements of VGG16, and text data is cleaned and structured.                                                                                 |
| **BLEU Score**         | Bilingual Evaluation Understudy Score, a metric for evaluating a generated sentence to a reference sentence, widely used in machine translation and text generation.           | Used to evaluate the quality of the generated captions in comparison with the actual captions.                                                                                             |
