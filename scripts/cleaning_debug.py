#!/usr/bin/env python

import os
import shutil


def main():
    dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation")
    for root, _, _ in os.walk(dir_):
        if "debug" in root:
            print("Removing directory: {root} ...".format(root=root))
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
