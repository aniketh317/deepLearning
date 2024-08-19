import tensorflow as tf

l0 = [1,2,3]
l1 = [4,5,6,7]
l2 = [7,8,9]
l3 = [1,2]

feat = [l0,l1,l2,l3]
label = [1,0,1,1]

def dataset_generator(features, labels):
    def func():
        for i in range(len(features)):
            yield (features[i], labels[i])
    return func

dataset = tf.data.Dataset.from_generator(dataset_generator(feat, label),
                                         output_signature=(
                                             tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                             tf.TensorSpec(shape=(), dtype=tf.int32)
                                         ))

# Define a function to batch ragged tensors
def ragged_batch_fn2(feature, label):
    return tf.RaggedTensor.from_row_lengths(feature, [len(feature)]), label

def apply_func(dataset):
    import pdb; pdb.set_trace()
    return dataset

ragged_batch_dataset = dataset.map(ragged_batch_fn2).apply(apply_func)