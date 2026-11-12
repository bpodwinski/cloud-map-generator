//! Where the project's files are: the shipped reference map, the example bodies, the default output folders.
//!
//! The Python tools find them from their own file (`HERE`). A binary has no such folder, so the root is, in
//! order: `--root` / `CLOUD_MAP_GENERATOR_ROOT`, the nearest ancestor of the working directory that holds the reference
//! map, the same search from the executable's folder, then from the folder this crate was built in; failing
//! all of them, the working directory (then a command that needs a shipped file says which one is missing).

use std::path::{Path, PathBuf};

/// `data/earth_cloud_fraction_bluemarble_2048.png`: NASA Blue Marble's cloud fraction, the reference of
/// `--match-earth` and `measure` (`measure.REFERENCE`).
pub const REFERENCE: &str = "data/earth_cloud_fraction_bluemarble_2048.png";

#[derive(Clone, Debug)]
pub struct Project {
    pub root: PathBuf,
}

fn ancestor_with_reference(start: &Path) -> Option<PathBuf> {
    start.ancestors().find(|d| d.join(REFERENCE).is_file()).map(Path::to_path_buf)
}

impl Project {
    /// The root, as described in the module comment.
    pub fn discover(explicit: Option<PathBuf>) -> Project {
        if let Some(root) = explicit {
            return Project { root };
        }
        let cwd = std::env::current_dir().ok();
        let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
        let built = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        let root = [cwd.clone(), exe, Some(built)]
            .into_iter()
            .flatten()
            .find_map(|d| ancestor_with_reference(&d))
            .or(cwd)
            .unwrap_or_else(|| PathBuf::from("."));
        Project { root }
    }

    /// The reference map (`measure.REFERENCE`).
    pub fn reference(&self) -> PathBuf {
        self.root.join(REFERENCE)
    }

    /// `cloudgen/bodies.py` `DEFAULT_BODIES`.
    pub fn default_bodies(&self) -> PathBuf {
        self.root.join("examples").join("bodies.json")
    }

    /// `output/`: the run artifacts (gitignored).
    pub fn output(&self) -> PathBuf {
        self.root.join("output")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_root_wins_and_paths_hang_off_it() {
        let p = Project::discover(Some(PathBuf::from("some/root")));
        assert_eq!(p.root, PathBuf::from("some/root"));
        assert_eq!(p.reference(), PathBuf::from("some/root").join(REFERENCE));
        assert_eq!(p.default_bodies(), PathBuf::from("some/root/examples/bodies.json"));
        assert_eq!(p.output(), PathBuf::from("some/root/output"));
    }

    #[test]
    fn a_folder_with_the_reference_is_found_from_below() {
        let dir = std::env::temp_dir().join(format!("cloud-map-generator-cli-root-{}", std::process::id()));
        let deep = dir.join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::create_dir_all(dir.join("data")).unwrap();
        std::fs::write(dir.join(REFERENCE), b"x").unwrap();
        assert_eq!(ancestor_with_reference(&deep), Some(dir.clone()));
        assert_eq!(ancestor_with_reference(&dir.join("a")), Some(dir));
    }
}
