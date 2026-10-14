"""Read MMseqs representative/member assignments without reversing columns."""
from pathlib import Path


def read_clusters(path):
    members = {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not all(fields):
                raise ValueError("Malformed cluster row %d" % number)
            representative, member = fields
            if member in members and members[member] != representative:
                raise ValueError("Conflicting cluster assignments for " + member)
            members[member] = representative
    if not members or any(members.get(rep) != rep for rep in set(members.values())):
        raise ValueError("Cluster representatives must have self membership")
    return members
