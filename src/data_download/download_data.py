import kagglehub


def fpb_download():
    # Download latest version
    path = kagglehub.dataset_download("n107hoangtuong/financial-phrasebank")

    print("Path to dataset files:", path)


def main():
    fpb_download()

if __name__ == "__main__":
    main()
