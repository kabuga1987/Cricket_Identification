from classification_helper import*
#==============================================

def run_classification_pipeline(
    p_segments,
    p_train_val_test,
    p_predictions,
    path_array,
    path_weights,
    shp,
    lr,
    fine_lr,
    epochs,
    batch_size,
    use_meanvar,
):

    os.makedirs(p_predictions, exist_ok=True)
    os.makedirs(os.path.dirname(path_weights), exist_ok=True)

    encoder = LabelEncoder(p_segments, p_train_val_test)

    train_x, train_y, val_x, val_y, test_x, test_y, n2_x, n2_y, n3_x, n3_y = encoder.execution()

    tgen = Generator(train_x, train_y, shp, batch_size, path_array, use_meanvar)
    vgen = Generator(val_x, val_y, shp, batch_size, path_array, use_meanvar)
    tst_gen = Generator(test_x, test_y, shp, batch_size, path_array, use_meanvar)
    n2_gen = Generator(n2_x, n2_y, shp, batch_size, path_array, use_meanvar)
    n3_gen = Generator(n3_x, n3_y, shp, batch_size, path_array, use_meanvar)

    trainer = Training()
    trainer.architecture = ModelArchitecture(shp)
    trainer.Tgenerator = tgen
    trainer.Vgenerator = vgen

    model = trainer.train(lr, fine_lr, path_weights, epochs)

    predictor = Predictions(
        model=model,
        p_preds=p_predictions,
        data={
            "one_n1_val": (val_x, val_y, vgen),
            "one_n1_test": (test_x, test_y, tst_gen),
            "one_n2_test": (n2_x, n2_y, n2_gen),
            "one_n3_test": (n3_x, n3_y, n3_gen),
        }
    )

    return predictor.execution()


if __name__ == "__main__":

    P_SEGMENTS = "../segment_index_extraction/segment_data.csv"
    P_TRAIN_VAL_TEST = "train_val_test_segment_data/"
    P_PREDICTIONS = "classification_predictions/CNNs/"
    PATH_ARRAY = "../segment_spectrogram_mfcc_feature_extraction/spectrogram_arrays/"
    PATH_WEIGHTS = "weights/n1_classification.h5"

    SHP = (128, 173, 1)
    LR = 1e-3
    FINE_LR = 1e-4
    EPOCHS = 1
    BATCH_SIZE = 128
    USE_MEANVAR = True

    metrics = run_classification_pipeline(
        p_segments=P_SEGMENTS,
        p_train_val_test=P_TRAIN_VAL_TEST,
        p_predictions=P_PREDICTIONS,
        path_array=PATH_ARRAY,
        path_weights=PATH_WEIGHTS,
        shp=SHP,
        lr=LR,
        fine_lr=FINE_LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_meanvar=USE_MEANVAR,
    )

    print(f"\n{metrics}")