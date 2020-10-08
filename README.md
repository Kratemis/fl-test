# fl-test

* --s3-client-models-folder (S3 folder for client models)
* --s3-main-models-folder (S3 folder for main models)
* --main-model (S3 folder for main models)
* --main-bucket (Bucket name for main models)
* --clients-bucket (Bucket name for client models)
* --local-models-folder (Local folder for client models)
* --s3-access-key (Credentials for AWS)
* --s3-secret-key (Credentials for AWS)


## Example command

```bash
python main.py --s3-client-models-folder clients --s3-main-models-folder main --main-model "main_model.pt" --client-models "1602148124_model.pt,1602148331_model.pt,1602148419_model.pt" --clients-bucket "MY_BUCKET" --main-bucket "ANOTHER_BUCKET_NAME" --local-models-folder "./storage"
```