import io
import sqlite3
from sqlite3 import Connection

import pandas as pd

from prototype.db.batch_fetch_images import IMAGE_COLUMNS, batch_fetch_images


def create_table_and_ingest_data(df: pd.DataFrame, table_name: str, conn: Connection):
    cursor = conn.cursor()

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        productId INTEGER PRIMARY KEY,
        gender TEXT,
        description TEXT,
        imageURL1 TEXT,
        imageURL2 TEXT,
        imageURL3 TEXT,
        imageURL4 TEXT,
        image1 BLOB,
        image2 BLOB,
        image3 BLOB,
        image4 BLOB,
        name TEXT,
        productType TEXT,
        pattern TEXT,
        productIdentifier TEXT
    )
    """
    cursor.execute(create_table_query)

    insert_query = f"""
    INSERT OR IGNORE INTO {table_name}(
        productId, gender, description, imageURL1, imageURL2, imageURL3, imageURL4,
        image1, image2, image3, image4, name, productType, pattern, productIdentifier
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    data_to_insert = []
    for _, row in df.iterrows():
        # Open image files and convert to bytes
        image_blobs = []
        for i in range(1, len(IMAGE_COLUMNS) + 1):
            image = row[f"image{i}"]
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")  # Convert image to bytes
                image_bytes = buffer.getvalue()
                image_blobs.append(sqlite3.Binary(image_bytes))

        data_to_insert.append(
            (
                row["productId"],
                row["gender"],
                row["description"],
                row["imageURL1"],
                row["imageURL2"],
                row["imageURL3"],
                row["imageURL4"],
                image_blobs[0],
                image_blobs[1],
                image_blobs[2],
                image_blobs[3],
                row["name"],
                row["productType"],
                row["pattern"],
                row["productIdentifier"],
            )
        )
    try:
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        print("Data inserted successfully.")
    except sqlite3.InternalError as e:
        print(f"Failed to insert data: {e}")


if __name__ == "__main__":
    # Connect to DB
    # TODO these should be in config and sourced from there
    db_file = "./database.db"
    db_conn = sqlite3.connect(db_file)
    db_table_name = "train"

    # Load data
    data = pd.read_csv("../../data/exercise_train.tsv", sep="\t")

    # Fetch images
    data = batch_fetch_images(data)

    # Insert in db
    create_table_and_ingest_data(data, db_table_name, db_conn)
    db_conn.close()
