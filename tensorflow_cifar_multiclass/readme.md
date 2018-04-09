# Tensorflow CIFAR10 multiclass

This is an adaptation of the CIFAR10 CNN model to allow for training examples having
multiple labels.


softmax 




#Using with AWS


## Transfer data to provisioned machine from S3
``` bash
aws configure
aws s3 sync s3://bucketname bucketdir/
```

```bash
python train.py --train_dir ~/planet_trn/ --data_dir ~/planet
```

