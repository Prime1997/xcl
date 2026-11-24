//! Cache writes normally use an existing directory. Only a missing path needs creation.

use std::io;
use std::path::Path;

/// Write a cache's temporary file; recover a missing parent once, without remembering
/// directory existence. Callers keep their existing atomic publication and best-effort rules.
pub(super) fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    match std::fs::write(path, bytes) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let Some(parent) = path.parent() else { return Err(error) };
            std::fs::create_dir_all(parent)?;
            std::fs::write(path, bytes)
        }
        result => result,
    }
}
