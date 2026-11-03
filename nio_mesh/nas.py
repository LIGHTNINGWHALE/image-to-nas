from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Node:
    node_id: int
    x: float
    y: float
    z: float = 0.0
    pixel_x: float | None = None
    pixel_y: float | None = None


@dataclass(slots=True)
class Element:
    card: str
    element_id: int
    prop_id: int
    node_ids: list[int]


@dataclass(slots=True)
class NasMesh:
    nodes: list[Node]
    elements: list[Element]
    metadata: dict = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        counter = Counter(element.card for element in self.elements)
        counter["GRID"] = len(self.nodes)
        return dict(counter)


def compact_mesh(
    mesh: NasMesh,
    *,
    drop_orphan_nodes: bool = True,
    renumber_nodes: bool = True,
    renumber_elements: bool = False,
) -> NasMesh:
    referenced_node_ids = {node_id for element in mesh.elements for node_id in element.node_ids}
    nodes = [
        node
        for node in mesh.nodes
        if (not drop_orphan_nodes) or node.node_id in referenced_node_ids
    ]

    if renumber_nodes:
        node_id_map = {node.node_id: index for index, node in enumerate(nodes, start=1)}
    else:
        node_id_map = {node.node_id: node.node_id for node in nodes}

    compact_nodes = [
        Node(
            node_id=node_id_map[node.node_id],
            x=node.x,
            y=node.y,
            z=node.z,
            pixel_x=node.pixel_x,
            pixel_y=node.pixel_y,
        )
        for node in nodes
    ]

    compact_elements: list[Element] = []
    for index, element in enumerate(mesh.elements, start=1):
        if not all(node_id in node_id_map for node_id in element.node_ids):
            continue
        compact_elements.append(
            Element(
                card=element.card,
                element_id=index if renumber_elements else element.element_id,
                prop_id=element.prop_id,
                node_ids=[node_id_map[node_id] for node_id in element.node_ids],
            )
        )

    metadata = dict(mesh.metadata)
    metadata["node_compacted"] = bool(drop_orphan_nodes or renumber_nodes)
    metadata["original_node_count"] = len(mesh.nodes)
    metadata["compacted_node_count"] = len(compact_nodes)
    return NasMesh(nodes=compact_nodes, elements=compact_elements, metadata=metadata)


def _format_small_field(value: float | int | str | None) -> str:
    if value is None:
        return " " * 8
    if isinstance(value, str):
        return f"{value:<8}"[:8]
    if isinstance(value, int):
        return f"{value:>8d}"[-8:]

    for precision in (5, 4, 3, 2, 1, 0):
        text = f"{value:.{precision}f}"
        if len(text) <= 8:
            return f"{text:>8}"
    text = f"{value:.1e}"
    return f"{text:>8}"[-8:]


def _format_card(card: str, fields: list[float | int | str | None]) -> str:
    return "".join([f"{card:<8}"] + [_format_small_field(field) for field in fields]).rstrip()


def parse_nas(path: str | Path) -> NasMesh:
    path = Path(path)
    nodes: list[Node] = []
    elements: list[Element] = []

    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.startswith("BEGIN") or raw_line.startswith("ENDDATA"):
            continue
        fields = [raw_line[idx : idx + 8].strip() for idx in range(0, len(raw_line), 8)]
        card = fields[0]
        if card == "GRID":
            nodes.append(
                Node(
                    node_id=int(fields[1]),
                    x=float(fields[3]),
                    y=float(fields[4]),
                    z=float(fields[5] or 0.0),
                )
            )
        elif card in {"CQUAD4", "CTRIA3"}:
            payload = [field for field in fields[1:] if field]
            elements.append(
                Element(
                    card=card,
                    element_id=int(payload[0]),
                    prop_id=int(payload[1]),
                    node_ids=[int(field) for field in payload[2:]],
                )
            )

    return NasMesh(nodes=nodes, elements=elements, metadata={"source_path": str(path)})


def write_nas(mesh: NasMesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["BEGIN BULK"]
    for node in mesh.nodes:
        lines.append(_format_card("GRID", [node.node_id, None, node.x, node.y, node.z]))
    for element in mesh.elements:
        fields: list[float | int | str | None] = [element.element_id, element.prop_id, *element.node_ids]
        lines.append(_format_card(element.card, fields))
    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n")
    return path
