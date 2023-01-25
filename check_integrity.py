from pathlib import Path
from tqdm import tqdm


def main():
    dataset_path = Path('/mnt/beegfs/work/FoMo_AIISDH/datasets/font_square')
    dirs = [x for x in dataset_path.iterdir() if x.is_dir()]
    incomplete_folders = []
    for dir in tqdm(dirs):
        files = [x for x in dir.rglob('*') if x.is_file()]
        num_files = len(files)
        # print(f'Checking {dir}... The folder contains {num_files} files.')
        if num_files != 300:
            print(f'The folder contains {num_files} files.')
            incomplete_folders.append(dir)

    print(f'The incomplete folders are {incomplete_folders}')


if __name__ == '__main__':
    main()
