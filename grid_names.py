"""Lists all grid names."""
from arct import DBI


for name in sorted(set(x['name'] for x in DBI.grid.all())):
    print(name)
