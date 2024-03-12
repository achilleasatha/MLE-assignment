import pandera

TRAINING_DATA_SCHEMA = pandera.DataFrameSchema(
    {
        "productId": pandera.Column(pandera.Int),
        "gender": pandera.Column(pandera.String),
        "description": pandera.Column(pandera.String),
        "imageURL1": pandera.Column(pandera.String),
        "imageURL2": pandera.Column(pandera.String),
        "imageURL3": pandera.Column(pandera.String),
        "imageURL4": pandera.Column(pandera.String),
        "name": pandera.Column(pandera.String),
        "productType": pandera.Column(pandera.String),
        "pattern": pandera.Column(pandera.String),
        "productIdentifier": pandera.Column(pandera.String),
    }
)

TEST_DATA_SCHEMA = pandera.DataFrameSchema(
    {
        "productId": pandera.Column(pandera.Int),
        "gender": pandera.Column(pandera.String),
        "description": pandera.Column(pandera.String),
        "imageURL1": pandera.Column(pandera.String),
        "imageURL2": pandera.Column(pandera.String),
        "imageURL3": pandera.Column(pandera.String),
        "imageURL4": pandera.Column(pandera.String),
        "name": pandera.Column(pandera.String),
        "productType": pandera.Column(pandera.String),
        "productIdentifier": pandera.Column(pandera.String),
    }
)

INFERENCE_DATA_SCHEMA = pandera.SeriesSchema(
    pandera.String, index=pandera.Index(pandera.String, name="productIdentifier")
)
