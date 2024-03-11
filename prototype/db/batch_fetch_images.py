import logging
from io import BytesIO

import pandas as pd
import requests
from PIL import Image

IMAGE_COLUMNS = ["imageURL1", "imageURL2", "imageURL3", "imageURL4"]


def batch_fetch_images(df: pd.DataFrame) -> pd.DataFrame:
    # TODO add PyDantic BaseModels for input output dfs
    logger = logging.getLogger(__name__)

    for col_index, col in enumerate(IMAGE_COLUMNS, 1):
        fetched_images = []
        for url in df[col]:
            try:
                response = requests.get(f"https://{url}", timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                fetched_images.append(image)
            except requests.HTTPError as e:
                logger.error(f"HTTP error occurred for image {url}: {e}")
                fetched_images.append(None)  # Append None if image fetching fails
        df = df.assign(**{f"image{col_index}": fetched_images})
    return df

    # NOTE: I tried to do a batch fetch but fails with 403 this is a bit unfortunate as
    # spamming the server with individual requests to fetch all images isn't great...

    # image_urls: list[str] = []
    # for n, col in enumerate(IMAGE_COLUMNS):
    #     image_urls.extend([f'https://{url}' for url in df[col].values])
    #
    #     response = requests.get(','.join(image_urls))
    #     try:
    #         response.raise_for_status()
    #     except requests.HTTPError as e:
    #         logger.error(f"HTTP error occurred: {e}")
    #         exit(1)
    #     else:
    #         fetched_images: list[Image] = []
    #         for url, content in zip(image_urls, response.iter_lines()):
    #             fetched_images.append(Image.open(BytesIO(content)))
    #         df[f'image{n+1}'] = fetched_images
    #     return df
