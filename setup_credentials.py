"""Asks for the R2 credentials the training needs and saves them in r2_credentials.yaml (gitignored).

    python setup_credentials.py            # writes r2_credentials.yaml
    python setup_credentials.py --exports  # prints `export` lines instead, to paste in a shell or ~/.bashrc
    python setup_credentials.py --hidden   # hides the keys while typing (no arrow key editing then)

An existing value is shown ready to edit: move with the arrow keys, or just press enter to keep it. The reader (common/training_parameters_reader.py) also
accepts the same names as environment variables, and the file path can be changed with R2_CREDENTIALS_FILE.
"""
import argparse
import getpass
import os
import readline  # noqa: F401, makes input() editable with the arrow keys
import shlex

import yaml

# (key, label, secret)
FIELDS = [
    ('r2_endpoint', 'R2 endpoint (https://<account>.r2.cloudflarestorage.com)', False),
    ('r2_access_key', 'R2 access key', True),
    ('r2_secret_key', 'R2 secret key', True),
    ('r2_bucket_name', 'R2 bucket name', False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exports', action='store_true', help='print export lines instead of writing the file')
    parser.add_argument('--hidden', action='store_true', help='do not echo the keys (they cannot be edited)')
    args = parser.parse_args()

    path = os.environ.get('R2_CREDENTIALS_FILE', 'r2_credentials.yaml')
    current = {}
    if os.path.exists(path):
        with open(path) as f:
            current = yaml.safe_load(f) or {}

    values = {}
    for key, label, secret in FIELDS:
        existing = os.environ.get(key.upper()) or current.get(key)
        if secret and args.hidden:
            answer = getpass.getpass(f'{label}{" [enter keeps current]" if existing else ""}: ').strip()
        else:
            # the current value is put in the line buffer, so it can be edited instead of retyped
            readline.set_startup_hook(lambda: readline.insert_text(existing or ''))
            try:
                answer = input(f'{label}: ').strip()
            finally:
                readline.set_startup_hook()
        values[key] = answer or existing
        if not values[key]:
            raise SystemExit(f'{key} is required')

    if args.exports:
        for key, value in values.items():
            print(f'export {key.upper()}={shlex.quote(value)}')
        return

    # created private from the start, the file holds secrets
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(values, f, default_flow_style=False)
    os.chmod(path, 0o600)
    print(f'saved to {path}')


if __name__ == '__main__':
    main()
