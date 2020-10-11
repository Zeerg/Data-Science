Mostly taken from Malware Science Book and updated some things

https://www.malwaredatascience.com/


```python
from keras.models import Model
from keras import layers


def my_model_simple(input_length=1024):
    input = layers.Input(shape=(input_length,), dtype='float32')
    middle = layers.Dense(units=512, activation='relu')(input)
    output = layers.Dense(units=1, activation='sigmoid')(middle)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def my_model(input_length=1024):
    # Note that we can name any layer by passing it a "name" argument.
    input = layers.Input(shape=(input_length,), dtype='float32', name='input')

    # We stack a deep densely-connected network on tops
    x = layers.Dense(2048, activation='relu')(input)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)

    # And finally we add the last (logistic regression) layer:
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
```


```python
def roc_plot(fpr, tpr, path_to_file):
    """
    :param fpr: array of false positive rates (an output from metrics.roc_curve())
    :param tpr: array of true positive rates (an output from metrics.roc_curve())
    :param path_to_file: where you wish to save the .png file
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.grid(True)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.ylim([0, 1])

    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)

    plt.semilogx(fpr, tpr, 'b-', label="Test set")
    plt.savefig(path_to_file)
    fig.clear()
    plt.close(fig)
```


```python
import numpy as np
from keras import callbacks
from sklearn import metrics


class MyCallback(callbacks.Callback):
    """
    Custom Keras callback to print validation AUC metric during training.
    Allowable over-writable methods:
    on_epoch_begin, on_epoch_end, on_batch_begin, on_batch_end,
    on_train_begin, on_train_end
    """

    def on_epoch_end(self, epoch, logs={}):
        validation_labels = self.validation_data[1]
        validation_scores = self.model.predict(self.validation_data[0])
        # flatten the scores:
        validation_scores = [el[0] for el in validation_scores]
        fpr, tpr, thres = metrics.roc_curve(y_true=validation_labels,
                                            y_score=validation_scores)
        auc = metrics.auc(fpr, tpr)
        print('\n\tEpoch {}, Validation AUC = {}'.format(epoch,
                                                         np.round(auc, 6)))
```


```python
from keras.models import load_model
import numpy as np
import mmh3
import re
import os


def read_file(sha, dir):
    with open(os.path.join(dir, sha), 'r', encoding='latin1') as fp:
        file = fp.read()
    return file


def extract_features(sha, path_to_files_dir,
                     hash_dim=1024, split_regex=r"\s+"):
    # first, read in the file as a big string:
    file = read_file(sha=sha, dir=path_to_files_dir)
    # next, split the big string into a bunch of different tokens ("words"):
    tokens = re.split(pattern=split_regex, string=file)
    # now take the module(hash of each token) so that each token is replaced
    # by bucket (category) from 1:hash_dim.
    token_hash_buckets = [
        (mmh3.hash(w) % (hash_dim - 1) + 1) for w in tokens
    ]
    # Finally, we'll count how many hits each bucket got, so that our features
    # always have length hash_dim, regardless of the size of the HTML file:
    token_bucket_counts = np.zeros(hash_dim)
    # this returns the frequency counts for each unique value in
    # token_hash_buckets:
    buckets, counts = np.unique(token_hash_buckets, return_counts=True)
    # and now we insert these counts into our token_bucket_counts object:
    for bucket, count in zip(buckets, counts):
        token_bucket_counts[bucket] = count
    return np.array(token_bucket_counts)


def my_generator(benign_files, malicious_files,
                 path_to_benign_files, path_to_malicious_files,
                 batch_size, features_length=1024):
    n_samples_per_class = batch_size / 2
    assert len(benign_files) >= n_samples_per_class
    assert len(malicious_files) >= n_samples_per_class
    while True:
        # first, extract features for some random benign files:
        ben_features = [
            extract_features(sha, path_to_files_dir=path_to_benign_files,
                             hash_dim=features_length)
            for sha in np.random.choice(benign_files, int(n_samples_per_class),
                                        replace=False)
        ]
        # now do the same for some malicious files:
        mal_features = [
            extract_features(sha, path_to_files_dir=path_to_malicious_files,
                             hash_dim=features_length)
            for sha in np.random.choice(malicious_files, int(n_samples_per_class),
                                        replace=False)
        ]
        # concatenate these together to get our features and labels array:
        all_features = ben_features + mal_features
        # "0" will represent "benign", and "1" will represent "malware":
        labels = [0 for i in range(int(n_samples_per_class))] + [1 for i in range(int(
            n_samples_per_class))]

        # finally, let's shuffle the labels and features so that the ordering
        # is not always benign, then malware:
        idx = np.random.choice(range(batch_size), batch_size)
        all_features = np.array([np.array(all_features[i]) for i in idx])
        labels = np.array([labels[i] for i in idx])
        yield all_features, labels


def make_training_data_generator(features_length, batch_size):
    path_to_training_benign_files = 'malware_data_science/ch11/chapter_11_UNDER_40/data/html/benign_files/training/'
    path_to_training_malicious_files = 'malware_data_science/ch11/chapter_11_UNDER_40/data/html/malicious_files/training/'

    train_benign_files = os.listdir(path_to_training_benign_files)
    train_malicious_files = os.listdir(path_to_training_malicious_files)

    training_generator = my_generator(
        benign_files=train_benign_files,
        malicious_files=train_malicious_files,
        path_to_benign_files=path_to_training_benign_files,
        path_to_malicious_files=path_to_training_malicious_files,
        batch_size=batch_size,
        features_length=features_length
    )
    return training_generator


def get_validation_data(features_length, n_validation_files):
    path_to_validation_benign_files = 'malware_data_science/ch11/chapter_11_UNDER_40/data/html/benign_files/validation/'
    path_to_validation_malicious_files = 'malware_data_science/ch11/chapter_11_UNDER_40/data/html/malicious_files/validation/'
    # get the validation keys:
    val_benign_files = os.listdir(path_to_validation_benign_files)
    val_malicious_files = os.listdir(path_to_validation_malicious_files)

    # create the model:
    # grab the validation data and extract the features:
    validation_data = my_generator(
        benign_files=val_benign_files,
        malicious_files=val_malicious_files,
        path_to_benign_files=path_to_validation_benign_files,
        path_to_malicious_files=path_to_validation_malicious_files,
        batch_size=n_validation_files,
        features_length=features_length
    ).__next__()
    return validation_data


def example_code_with_validation_data(model, training_generator, steps_per_epoch, features_length, n_validation_files):
    validation_data = get_validation_data(features_length, n_validation_files)
    model.fit_generator(
        validation_data=validation_data,
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        verbose=1)

    return model


features_length = 1024
# by convention, num_obs_per_epoch should be roughly equal to the size
# of your training dataset, but we're making it small here since this
# is example code and we want it to run fast!
num_obs_per_epoch = 500000
batch_size = 8000

# create the model using the function from the model architecture section:
model = my_model(input_length=features_length)

# make the training data generator:
training_generator = make_training_data_generator(batch_size=batch_size, features_length=features_length)
# and now train the model:
model.fit(training_generator, steps_per_epoch=num_obs_per_epoch / batch_size, epochs=10, workers=4, use_multiprocessing=True)

# Get validation or unseen dat
validation_data = get_validation_data(n_validation_files=100, features_length=1024)
validation_labels = validation_data[1]
validation_scores = [el[0] for el in model.predict(validation_data[0])]
# Evaluate the model with unseen data
print(validation_scores)

# save the model
model.save('my_model.h5')
# load the model back into memory from the file:
#same_model = load_model('my_model.h5')  # from keras.models.load_model
```

    Epoch 1/10
    WARNING:tensorflow:multiprocessing can interact badly with TensorFlow, causing nondeterministic deadlocks. For high performance data pipelines tf.data is recommended.
    7/6 [=================================] - 18s 3s/step - loss: 0.5492 - accuracy: 0.7506
    Epoch 2/10
    7/6 [=================================] - 40s 6s/step - loss: 0.2099 - accuracy: 0.9126
    Epoch 3/10
    7/6 [=================================] - 41s 6s/step - loss: 0.1170 - accuracy: 0.9596
    Epoch 4/10
    7/6 [=================================] - 24s 3s/step - loss: 0.1138 - accuracy: 0.9573
    Epoch 5/10
    7/6 [=================================] - 22s 3s/step - loss: 0.0864 - accuracy: 0.9718
    Epoch 6/10
    7/6 [=================================] - 36s 5s/step - loss: 0.1034 - accuracy: 0.9628
    Epoch 7/10
    7/6 [=================================] - 35s 5s/step - loss: 0.0396 - accuracy: 0.9881
    Epoch 8/10
    7/6 [=================================] - 20s 3s/step - loss: 0.0578 - accuracy: 0.9812
    Epoch 9/10
    7/6 [=================================] - 19s 3s/step - loss: 0.0387 - accuracy: 0.9883
    Epoch 10/10
    7/6 [=================================] - 31s 4s/step - loss: 0.0278 - accuracy: 0.9923
    [0.5928085, 0.9893746, 0.08666548, 0.04018569, 0.015443385, 0.7545063, 0.9999971, 0.24936265, 0.8646091, 0.99387914, 0.9999896, 0.5584367, 0.5826191, 0.32542306, 0.09676108, 0.99999714, 0.10852009, 0.028113931, 0.004361123, 0.334956, 0.08666548, 0.028113931, 0.010056436, 0.0039224327, 0.0039224327, 0.7666678, 0.24895257, 0.9939281, 0.69319624, 0.0039224327, 0.38869655, 0.9999963, 0.889449, 0.9999896, 0.5755563, 0.96055233, 0.7541746, 0.9999859, 0.7978991, 0.23959291, 0.110082775, 0.99387914, 0.795136, 0.04727465, 0.015738934, 0.110082775, 0.014317721, 0.028113931, 0.015443385, 0.24936265, 0.7515396, 0.7978991, 0.23946488, 0.7349657, 0.99527884, 0.23959291, 0.7545063, 0.71172106, 0.032189667, 0.0070393085, 0.004361123, 0.8646091, 0.7666678, 0.9746903, 0.5826191, 0.9999963, 0.23946488, 0.9939281, 0.9999971, 0.7349657, 0.010048389, 0.7978991, 0.013908386, 0.13377944, 0.5755563, 0.334956, 0.9939281, 0.889449, 0.87684685, 0.041002154, 0.010056436, 0.008945733, 0.008945733, 0.69319624, 0.3215278, 0.0039224327, 0.50521547, 0.795136, 0.008742899, 0.99999106, 0.99999714, 0.010752499, 0.013754249, 0.9999963, 0.7666678, 0.0039224327, 0.23946488, 0.72279775, 0.046046615, 0.889449]



```python
fpr, tpr, thres = metrics.roc_curve(y_true=validation_labels, y_score=validation_scores)
auc = metrics.auc(fpr, tpr)
print('Validation AUC = {}'.format(auc))
```

    Validation AUC = 0.8771508603441377


References

* https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab
* https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
* https://keras.io/api/optimizers/sgd/
* https://arxiv.org/abs/1412.6980 - Adam Algorithm


```python

```
