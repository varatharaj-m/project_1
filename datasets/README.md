# Datasets

Place only datasets whose terms permit your intended use here. Document source, license/terms, version, collection date, label definitions, and preprocessing steps. The GitHub projects below do not all contain redistributable datasets or complete model files; do not treat repository visibility as a dataset license.

## Referenced sources

- [Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification), referenced by [praj-tarun/cyberbullying-detection](https://github.com/praj-tarun/cyberbullying-detection). Obtain it from Kaggle and follow its current dataset terms. Expected fields include `tweet` and `cyberbullying_type`.
- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), also referenced by the project above. Kaggle account/competition terms may apply. Its `train.csv` commonly has `comment_text` and six binary toxicity labels.
- [HateMM](https://github.com/loris-fab/Deep_learning) is cited by the multimodal hate-speech project; the dataset is not bundled in that repository. Request/access it under the dataset authors' terms before using it.
- [am610/DeepFake](https://github.com/am610/DeepFake) describes a locally held set of real/fake videos, not a dataset download. Its README documents frame-level split leakage; do not reuse its reported held-out score as a reliable video-level estimate.

The CSVs in `praj-tarun/cyberbullying-detection/data/raw/` are Git LFS objects rather than ordinary CSV payloads in the raw GitHub view. The pretrained weights in `Dipanshu7777/bullying-detector` are also an LFS object, and that repository currently has no standalone `LICENSE` file despite the README's MIT claim. Neither is copied into this project.

Suggested schemas:

- Text safety: `text,label` or `text,labels` for multi-label data.
- Fake news: `title,text,label,source_url,published_at`.
- Image: `path,label,source,dataset_split`.
- Video: `path,label,group_id` where `group_id` prevents frame leakage across splits.
- Audio: `path,transcript,label,speaker_or_group_id`.

## OpenFake image dataset

[ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) is an image dataset with `image` and `label` (`real`/`fake`) fields and `core` and `reddit` configurations. Its license is **CC BY-NC 4.0**: it is for non-commercial use only. The complete v2 dataset is about 3.4 TB, so do not clone/download the full repository. The training script streams a bounded sample from Hugging Face and pins the dataset revision; it does not store the original dataset under this repository.

```powershell
python -m ml.training.train_openfake_image_model --samples-per-class 256 --validation-per-class 64 --epochs 3 --evaluate-ood
```

The optional OOD evaluation reports the provided `core/test` and `reddit/test` splits separately. After training, set `IMAGE_MODEL_PATH=model_weights/openfake_image.pt` and `ALLOW_NONCOMMERCIAL_MODELS=true` in `.env`, then restart the app. The second setting is an explicit runtime opt-in; without it, OpenFake-derived weights stay unavailable. The model and any weights trained from OpenFake inherit its non-commercial-use restriction; they must not be deployed commercially. Keep the adjacent JSON metadata and attribution with the model artifact.

## Training the text classifier

Put an authorized CSV under `datasets/`, then train using a multiclass label column:

```powershell
python -m ml.training.train_text_model --dataset datasets/cyberbullying_tweets.csv --text-column tweet --label-column cyberbullying_type
```

For Jigsaw-style multi-label CSVs, specify the six binary columns:

```powershell
python -m ml.training.train_text_model --dataset datasets/jigsaw_train.csv --text-column comment_text --label-columns toxic severe_toxic obscene threat insult identity_hate
```

Training writes `model_weights/text_classifier.joblib` and a JSON metadata file beside it. Set `TEXT_MODEL_PATH` to that artifact path before starting the app to use it; otherwise the transparent keyword baseline remains active. Validation metrics are generated from the supplied data and are not comparable across datasets without a shared evaluation protocol.

