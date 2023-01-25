from pathlib import Path
from tqdm import tqdm


def main():
    dataset_path = Path('/mnt/beegfs/work/FoMo_AIISDH/datasets/font_square')
    num_fonts_per_split = 104
    dirs = sorted([x for x in dataset_path.iterdir() if x.is_dir()])

    num_splits = len(dirs) // num_fonts_per_split

    for i in tqdm(range(num_splits)):
        start_idx = i*num_fonts_per_split
        current_dirs = dirs[start_idx:start_idx+num_fonts_per_split]
        split_path = Path(dataset_path, f'{i*104:05d}')
        split_path.mkdir()
        for current_dir in current_dirs:
            current_dir.rename(Path(split_path, current_dir.stem))

    print('Done!')


if __name__ == '__main__':
    main()