from pretraining_classification_helper import *

#===============================================

def run_pretrained_classification_pipeline(
    p_segments,
    p_train_val_test,
    p_predictions,
    path_array,
    path_weights,
    shp,
    lr_start,
    lr_end,
    epochs,
    batch_size,
    use_pretrained,
    fine_tune,
):

    os.makedirs(p_predictions, exist_ok=True)
    os.makedirs(os.path.dirname(path_weights), exist_ok=True)

    encoder = LabelEncoder(p_segments, p_train_val_test)

    train_x, train_y, val_x, val_y, test_x, test_y, n2_x, n2_y, n3_x, n3_y = encoder.execution()

    tgen = Generator(train_x, train_y, shp, batch_size, path_array)
    vgen = Generator(val_x, val_y, shp, batch_size, path_array)
    tst_gen = Generator(test_x, test_y, shp, batch_size, path_array)
    n2_gen = Generator(n2_x, n2_y, shp, batch_size, path_array)
    n3_gen = Generator(n3_x, n3_y, shp, batch_size, path_array)

    n_classes = train_y[0].shape[0]

    architecture = CricketClassifier(
        input_shape=shp,
        num_classes=n_classes,
        use_pretrained=use_pretrained,
        fine_tune=fine_tune,
    )

    trainer = Training()
    trainer.architecture = architecture
    trainer.Tgenerator = tgen
    trainer.Vgenerator = vgen

    model = trainer.train(lr_start, lr_end, path_weights, epochs)

    predictor = Predictions(
        model=model,
        p_preds=p_predictions,
        data={
            "validation": (val_x, val_y, vgen),
            "one_n1_test": (test_x, test_y, tst_gen),
            "one_n2_test": (n2_x, n2_y, n2_gen),
            "one_n3_test": (n3_x, n3_y, n3_gen),
        }    
    )

    return predictor.execution()


if __name__ == "__main__":

    # Paths
    P_SEGMENTS = "../segment_index_extraction/segment_data.csv"
    P_TRAIN_VAL_TEST = "../closed_population_ID/train_val_test_segment_data/"
    P_PREDICTIONS = "classification_predictions/"
    PATH_ARRAY = "../segment_spectrogram_mfcc_feature_extraction/spectrogram_arrays/"
    PATH_WEIGHTS = "weights/n1_pretraining_classification.h5"

    # Model / training configuration
    SHP = (224, 224, 3)
    LR_START = 1e-3
    LR_END = 1e-5
    EPOCHS = 1
    BATCH_SIZE = 128

    # Transfer learning configuration
    USE_PRETRAINED = True # set to False if you want full finetuning, i.e, training from scratch
    FINE_TUNE = True # Set to False if you want to re-use the pretrained weights

    metrics = run_pretrained_classification_pipeline(
        p_segments=P_SEGMENTS,
        p_train_val_test=P_TRAIN_VAL_TEST,
        p_predictions=P_PREDICTIONS,
        path_array=PATH_ARRAY,
        path_weights=PATH_WEIGHTS,
        shp=SHP,
        lr_start=LR_START,
        lr_end=LR_END,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_pretrained=USE_PRETRAINED,
        fine_tune=FINE_TUNE,
    )

    print(f"\n{metrics}")