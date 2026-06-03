from triplet_CNN_helper import*

#==============================

def run_matching_pipeline(
    p_train,
    p_validation,
    p_arrays,
    p_weights,
    p_preds,
    p_tests="pairs/",
    shp=(128, 173, 1),
    batch_size=256,
    samples_per_ID=16,
    epochs=10,
    lr_start=1e-3,
    lr_end=1e-5,
    meanvar=True,
):

    loader = DataLoader(p_train, p_validation, samples_per_ID)
    Ts, TYs, Vs, VYs = loader.train_validation_samples_labels()

    train_gen = Generator(Ts, TYs, shp, batch_size, p_arrays, meanvar)
    val_gen = Generator(Vs, VYs, shp, batch_size, p_arrays, meanvar)

    trainer = Training()
    trainer.architecture = ModelArchitecture(shp=shp)
    trainer.Tgenerator = train_gen
    trainer.Vgenerator = val_gen

    model = trainer.train(lr_start, lr_end, p_weights, epochs)

    evaluator = Evaluation()

    predictor = Predictions(
        path_val_test_samples=p_train.rsplit("/", 1)[0] + "/",
        path_val_samples=p_validation,
        path_val_pairs=f"{p_tests}n1_validation_pairs.csv",
        path_tests=p_tests,
        p_arrays=p_arrays,
        p_preds=p_preds,
        shp=shp,
        bs=batch_size,
    )

    predictor.model = model
    predictor.evaluator = evaluator

    df_metrics = predictor.execution(model)

    print("\n" + "=" * 50)
    print("Evaluation metrics")
    print("=" * 50 + "\n")

    return model, df_metrics

#======================================================

if __name__ == "__main__":

    model, df_metrics = run_matching_pipeline(
        p_train="../closed_population_ID/train_val_test_segment_data/n1_train.csv",
        p_validation="../closed_population_ID/train_val_test_segment_data/n1_validation.csv",
        p_arrays="../segment_spectrogram_mfcc_feature_extraction/spectrogram_arrays/",
        p_weights="weights/n1_matching.h5",
        p_preds="matching_predictions/CNNs/",
        shp=(128, 173, 1),
        batch_size=256,
        samples_per_ID=16,
        epochs=1,
        lr_start=1e-3,
        lr_end=1e-5,
        meanvar=True
    )

    print(df_metrics)