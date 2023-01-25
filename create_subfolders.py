from pathlib import Path
from tqdm import tqdm


def main():
    dataset_path = Path('/mnt/beegfs/work/FoMo_AIISDH/datasets/font_square')
    num_fonts_per_split = 104
    dirs = sorted([x for x in dataset_path.iterdir() if x.is_dir()])

    num_splits = len(dirs) // num_fonts_per_split

    for i in tqdm(range(num_splits)):
        current_dirs = dirs[i:i+num_fonts_per_split]
        split_path = Path(dataset_path, f'{i*100:05d}')
        split_path.mkdir()
        for dir in current_dirs:
            dir.rename(Path(split_path, dir.stem))

    print('Done!')


if __name__ == '__main__':
    main()