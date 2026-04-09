
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from commu_umap_support import default_paths, load_commu_loss_table


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a cached COMMU latent/loss table for the UMAP notebook.')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--max-files', type=int, default=None, help='Optional cap for quick experiments.')
    parser.add_argument('--force', action='store_true', help='Ignore an existing cache and rebuild it.')
    args = parser.parse_args()

    paths = default_paths(args.repo_root)
    table = load_commu_loss_table(
        paths.loss_dir,
        max_files=args.max_files,
        use_cache=not args.force,
        cache_path=paths.cache_table,
    )
    print(f'Rows: {len(table)}')
    print(f'Cache: {paths.cache_table}')
    print(table[['segment_id', 'track_role', 'track_role_grouped', 'final_loss', 'kl_chd', 'kl_rhy']].head().to_string(index=False))


if __name__ == '__main__':
    main()
