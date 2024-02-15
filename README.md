# Image Captioning with Deep Learning

## Abstract

This project presents a sophisticated image captioning system, employing an enhanced deep learning architecture that integrates convolutional and recurrent neural networks. At its core, the model leverages the VGG16 neural network for high-level feature extraction from images. This robust feature extraction process uses a pre-trained VGG16 model, modified by removing its top layer, to transform images into a 4096-dimensional feature vector.

In the realm of textual data processing, the captions associated with images are tokenized to form a vocabulary, which is then fed into an Embedding layer. This embedding layer is crucial for translating discrete words into meaningful vector representations.

The project introduces significant enhancements to the initial model architecture for improved learning capability and generalization. The image feature processing pathway includes a Dense layer with 512 units, increased dropout for regularization, and Batch Normalization for stabilizing the learning process. On the textual data side, a major improvement is the adoption of a Bidirectional LSTM layer with 512 units, which enhances the model's ability to capture context from both directions of the text sequence, augmenting the capability of the standard LSTM layers. An additional LSTM layer and Batch Normalization are also included to further refine the text processing.

The model's decoder section combines the outputs of the image processing and text processing pathways. It features increased dense layer units and Batch Normalization, culminating in a Dense layer with softmax activation to generate the final caption outputs.

Overall, this advanced image captioning system demonstrates a harmonious integration of CNN for feature extraction and enhanced RNNs for sequential text processing, resulting in a robust model capable of generating contextually relevant and syntactically coherent captions for a wide array of images. The model's architecture, optimized with Batch Normalization and Bidirectional LSTM, marks a significant stride in the field of automated image captioning.

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
