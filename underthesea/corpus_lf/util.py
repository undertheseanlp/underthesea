from os import listdir
from os.path import join
from pathlib import Path


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


class FolderStructure:

    @staticmethod
    def show_folder_structure_incorrect(sample_path, current_path, root_sample_path, root_current_path):
        print("\n[IMPORT DATA ERROR]: Your folder structure is incorrect")
        short_sample_path = str(Path(sample_path))[len(str(Path(root_sample_path))):]
        if Path(sample_path).is_file():
            print(f"Cannot found file {short_sample_path}")
        if Path(sample_path).is_dir():
            print(f"Cannot found folder {short_sample_path}")

        print("\nCorrect folder structure")
        paths = DisplayablePath.make_tree(Path(root_sample_path))
        for path in paths:
            print(path.displayable())

        print("\nYour folder structure")
        paths = DisplayablePath.make_tree(Path(root_current_path))
        for path in paths:
            print(path.displayable())

        raise SystemExit("")

    @staticmethod
    def check_structure(sample_path, current_path, root_sample_path=None, root_current_path=None):
        if root_sample_path is None:
            root_sample_path = sample_path

        if root_current_path is None:
            root_current_path = current_path

        if (Path(sample_path).is_file() != Path(current_path).is_file()) or \
           (Path(sample_path).is_dir() != Path(current_path).is_dir()):
            FolderStructure.show_folder_structure_incorrect(sample_path, current_path, root_sample_path,
                                                            root_current_path)

        if Path(sample_path).is_dir():
            for sub_sample_path in listdir(Path(sample_path)):
                FolderStructure.check_structure(join(sample_path, sub_sample_path),
                                                join(current_path, sub_sample_path),
                                                root_sample_path=root_sample_path,
                                                root_current_path=root_current_path)
