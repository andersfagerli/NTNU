import pathlib
import zipfile
import requests
import tqdm

zip_url = "https://folk.ntnu.no/haakohu/tdt4265_2021_project/rdd2020.zip"

def download_image_zip(zip_path):
    response = requests.get(zip_url, stream=True)
    total_length = int(response.headers.get("content-length"))
    assert response.status_code == 200, \
        f"Did not download the images. Contact the TA. Status code: {response.status_code}"
    with open(zip_path, "wb") as fp:
        for data in tqdm.tqdm(
                response.iter_content(chunk_size=4096), total=int(total_length/4096)+1,
                desc="Downloading images."):
            fp.write(data)


if __name__ == "__main__":
    dataset_path = pathlib.Path("datasets", "RDD2020_filtered")
    if not dataset_path.parent.is_dir():
        dataset_path.parent.mkdir(exist_ok=True, parents=True)
    work_dataset_path = pathlib.Path("/work", "datasets", "RDD2020_filtered")
    if dataset_path.is_dir():
        print("Dataset already exists. If you want to download again, delete the folder", dataset_path.absolute())
        exit()
    if work_dataset_path.is_dir():
        print("You are working on a computer with the dataset under work_dataset.")
        print("We're going to copy all image files to your directory")
        print("Dataset setup finished. Extracted to:", dataset_path)

        dataset_path.symlink_to(work_dataset_path)
        exit()
    zip_path = dataset_path.parent.joinpath("rdd2020.zip")
    if not zip_path.is_file():
        download_image_zip(zip_path)
    print("Extracting dataset")
    with zipfile.ZipFile(zip_path, "r") as fp:
        fp.extractall(dataset_path.parent)
    print("Dataset setup finished. Extracted to:", dataset_path)
