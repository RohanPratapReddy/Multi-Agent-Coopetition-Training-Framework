# conf/db.yaml — owns: DATABASES. SQLite is the only supported engine; a
# non-sqlite ENGINE is refused by Settings.database_path.

DATABASES:
  default:
    ENGINE: sqlite3
    NAME: var/mactf.sqlite3
