//! The texts the server hands to the AI: its instructions and the two prompt templates.
//!
//! The wording is the Python server's, kept verbatim (it was refined by a blind usability test) in `src/text/`.
//! Generic on purpose: the reader may be building a game, a planetarium, a worldbuilding tool or a research
//! notebook.

/// What the server tells a client when it connects.
pub const INSTRUCTIONS: &str = include_str!("text/instructions.txt");

/// The resource `cloud-map-generator://docs/bodies-format`.
pub const BODIES_FORMAT: &str = include_str!("text/bodies_format.md");

const DESIGN_TEMPLATE: &str = include_str!("text/design_planet_clouds.txt");
const BAKE_TEMPLATE: &str = include_str!("text/bake_for_engine.txt");

/// `' '.join(text.split())[:limit] or fallback`: the argument on one line, shortened.
fn one_line(text: &str, limit: usize, fallback: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let cut: String = joined.chars().take(limit).collect();
    if cut.is_empty() {
        fallback.to_string()
    } else {
        cut
    }
}

/// The loop for designing a planet's cloud cover.
pub fn design_planet_clouds(planet_description: &str) -> String {
    DESIGN_TEMPLATE.replace("@@DESC@@", &one_line(planet_description, 2000, "(no description given)"))
}

/// The flow for baking the cloud maps of a bodies file.
pub fn bake_for_engine(bodies_file: &str) -> String {
    BAKE_TEMPLATE.replace("@@PATH@@", &one_line(bodies_file, 1000, "(no path given)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompts_render_their_argument_on_one_line() {
        let t = design_planet_clouds("a tidally   slow\nocean world");
        assert!(t.contains("\n\na tidally slow ocean world\n\n"), "{t}");
        assert!(design_planet_clouds("   ").contains("(no description given)"));
        assert!(design_planet_clouds(&"x ".repeat(5000)).len() < 6000);
        let b = bake_for_engine("bodies.json");
        assert!(b.contains("Bodies file: bodies.json\n") && b.contains("bake_bodies") && b.contains("url_prefix"));
        assert!(INSTRUCTIONS.starts_with("cloud-map-generator designs the global cloud cover") && INSTRUCTIONS.contains("cloud-map-generator://capabilities"));
        assert!(BODIES_FORMAT.contains("rotationPeriodDays") && BODIES_FORMAT.contains("rotation_period_days"));
    }
}
