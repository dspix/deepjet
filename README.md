# Fully convolutional net FCN8

Example

```python
import tensorflow as tf
# Add the import paths to the modules
import sys
sys.path.append('../data')
sys.path.append('../fcn8')
import fcn8_inputs as data
import net

tf.reset_default_graph()
layers = 3 # No. input layers
epochs = 1

# Input data are tensorflow records
samples = data.get_record_batches(
    'path to the records on file',
    1, # Batch size
    layers
)

# Build the model
iterator = tf.data.Iterator.from_structure(
    samples.output_types,
    samples.output_shapes
)

image, label = iterator.get_next()
model = net.inference(tf.cast(image, tf.float32), in_layers=layers)
batch_labels = net.batch_expand_labels(label, [0, 1, 2])
l = net.loss(model, batch_labels)
train_op = net.train(l, 1e-5) # Learning rate

with tf.Session() as sess:
    # Training loop
    for epoch in range(epochs):
        sess.run(iterator.make_initializer(samples))
        
        while True:
            try:
                loss_train = sess.run([train_op]) 
            except tf.errors.OutOfRangeError:
                break
                
```
