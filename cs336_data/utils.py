class DisjointSet:
    def __init__(self, elements):
        # parent dictionary maps element to its parent
        self.parent = {e: e for e in elements}
        # rank dictionary helps in balancing the tree during union
        self.rank = {e: 0 for e in elements}

    def find(self, item):
        """Finds the representative of the set containing item with path compression."""
        if self.parent[item] == item:
            return item
        # Path compression: make the node point directly to the root
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        """Unions two sets by rank."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            # Union by rank: attach smaller rank tree to larger rank tree
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            return True # Merged successfully
        return False # Already in the same set

