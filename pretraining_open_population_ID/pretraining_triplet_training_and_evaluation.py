from Pretraining_CNN_triplet_helper import*

#===========================================

def run_matching_pipeline(
    p_samples,
    p_arrays,
    p_weights,
    p_pairs,
    p_preds,
    shp,
    batch_size,
    samples_per_ID,
    epochs,
    lr_start,
    lr_end,
    meanvar
):

    # ==================================================
    # Data loading
    # ==================================================
    loader = DataLoader(
        path_train = os.path.join(p_samples, "n1_train.csv"),
        path_validation = os.path.join(p_samples, "n1_validation.csv"),
        samples_per_ID = samples_per_ID    
    ) 
                       
    Ts, TYs, Vs, VYs = loader.train_validation_samples_labels()

    # ==================================================
    # Generators
    # ==================================================
    Tgenerator = Generator(Ts, TYs, shp, batch_size, p_arrays)
    Vgenerator = Generator(Vs, VYs, shp, batch_size, p_arrays)

    # ==================================================
    # Model architecture
    # ==================================================
    architecture = CricketEmbedding(
        input_shape=shp,
        use_pretrained=False,
        fine_tune=True
    )

    # ==================================================
    # Training
    # ==================================================
    trainer = Training()
    trainer.architecture = architecture
    trainer.Tgenerator = Tgenerator
    trainer.Vgenerator = Vgenerator

    model, history = trainer.train(lr_start, lr_end, p_weights, epochs)

    # ==================================================
    # Evaluation
    # ==================================================
    evaluator = Evaluation()

    predictor = Predictions(
        p_samples, 
        p_pairs,
        p_arrays,
        p_preds,
        shp,
        batch_size
    )

    predictor.model = model
    predictor.evaluator = evaluator

    df_metrics = predictor.Execution()

    return model, df_metrics
#===========================
# RUN THE FULL EXPERIMENT
#===========================

if __name__ == "__main__":

    model, df_metrics = run_matching_pipeline(
        p_samples="../closed_population_ID/train_val_test_segment_data/",
        p_arrays="../segment_spectrogram_mfcc_feature_extraction/spectrogram_arrays/",
        p_weights="weights/n1_matching.h5",
        p_pairs="../open_population_ID/pairs/",
        p_preds="predictions/",
        shp=(224, 224, 3),
        batch_size=256,
        samples_per_ID=16,
        epochs=1,
        lr_start=1e-3,
        lr_end=1e-5,
        meanvar=True
    )

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50 )
    print(df_metrics)