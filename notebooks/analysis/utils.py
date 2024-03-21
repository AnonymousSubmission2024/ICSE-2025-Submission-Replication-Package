import dataclasses
from typing import overload


@dataclasses.dataclass
class PositionIndex:
    position_to_index: dict[tuple[int, int], int]
    index_to_position: dict[int, tuple[int, int]]

    @overload
    def __getitem__(self, index_or_position: tuple[int, int]) -> int:
        ...

    @overload
    def __getitem__(self, index_or_position: int) -> tuple[int, int]:
        ...

    def __getitem__(self, index_or_position: tuple[int, int] | int):
        if isinstance(index_or_position, tuple):
            return self.position_to_index[index_or_position]
        return self.index_to_position[index_or_position]

    def __contains__(self, index_or_position: tuple[int, int] | int) -> bool:
        if isinstance(index_or_position, tuple):
            return index_or_position in self.position_to_index

        return index_or_position in self.index_to_position

    @staticmethod
    def from_source(source: str):
        source_bytes = source.encode()
        source_lines = source_bytes.splitlines(keepends=True)
        position_to_index: dict[tuple[int, int], int] = {}
        index_to_position: dict[int, tuple[int, int]] = {}

        line_start_char_offset = 0

        for line_number, line in enumerate(source_lines):
            for col_number, _ in enumerate(line):
                curr_position = (line_number, col_number)
                curr_char_index = line_start_char_offset + col_number
                position_to_index[curr_position] = curr_char_index
                index_to_position[curr_char_index] = curr_position

            line_start_char_offset += len(line)

        return PositionIndex(position_to_index, index_to_position)


########## TODO add some tests like this to ensure PositionIndex is consistent with
########## tree-sitter's internal mapping of position/indices
# cst_nodes = list(
#     filter(
#     ts_utils.predicates.is_leaf,
#     ts_utils.iter.iternodes(tree.walk())
#     )
# )
# position_index = PositionIndex.from_source(source)

# for node in cst_nodes:
#     assert position_index[node.start_point] == node.start_byte
#     assert position_index[node.start_byte] == node.start_point
