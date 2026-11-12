//! What the binary carries: the files of `ui/static/`, embedded by `build.rs` (Earth's reference map is `cloud_map_generator_service::REFERENCE_PNG`).

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// The embedded file at `rel` (a path under `ui/static/`, forward slashes), if there is one.
pub fn static_file(rel: &str) -> Option<&'static [u8]> {
    STATIC_FILES.iter().find(|(name, _)| *name == rel).map(|(_, bytes)| *bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_and_its_script_are_embedded() {
        let index = std::str::from_utf8(static_file("index.html").expect("index.html")).unwrap().to_lowercase();
        assert!(index.contains("<html") || index.contains("<!doctype"));
        assert!(static_file("app.js").is_some_and(|js| !js.is_empty()));
        assert!(static_file("nope.js").is_none());
    }
}
