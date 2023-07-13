# torch-distributed-sandbox
Learning distributed in pytorch


`allreduce_toy.py` is an example of using allreduce for adding random numbers
`test_init.py` is an annotated test of whether distributed is working with many helpful links


## Testing OOM workaround

The following was tested on two RTX A5000s, with an MNIST image shape of 3000x3000

`mnist_onegpu.py` is a simple training script with MNIST for one GPU. With a batchsize of 10, it will OOM. I set the batchsize to 5 to work around this.

`mnist_distributed` is a distributed training script with MNIST for multiple GPUS. With a batchsize of 5, it does not OOM as expected from the one gpu experiment. Therefore, with two A5000s, we have an effective batch size of 10 samples. 