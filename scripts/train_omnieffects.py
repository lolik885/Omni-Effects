import os
import sys

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from cogvideox.train.schemas import parse_args
from cogvideox.train.trainer_omnieffects import Trainer


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.fit()


if __name__ == '__main__':
    main()
